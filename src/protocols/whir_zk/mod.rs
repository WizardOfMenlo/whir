mod committer;
mod config;
mod prover;
mod utils;
mod verifier;

pub use self::{
    committer::{Commitment, Witness},
    config::Config,
};

use ark_ff::{FftField, Field};

impl<F: FftField> Config<F> {
    /// Interleaving depth of the initial IRS commitment (= 2^folding_factor).
    pub(super) const fn interleaving_depth(&self) -> usize {
        self.blinded_commitment.initial_committer.interleaving_depth
    }

    /// Generator ω of the full NTT domain (size = num_rows × interleaving_depth).
    pub(super) fn omega_full(&self) -> F::BasePrimeField {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        let full_domain_size = num_rows * self.interleaving_depth();
        crate::algebra::ntt::generator(full_domain_size)
            .expect("full IRS domain should have primitive root")
    }

    /// Sub-domain generator (ω_sub = ω^interleaving_depth).
    fn omega_sub(&self) -> F::BasePrimeField {
        self.blinded_commitment.initial_committer.generator()
    }

    /// ζ = ω^num_rows — the interleaving_depth-th root of unity.
    pub(super) fn zeta(&self) -> F::BasePrimeField {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        self.omega_full().pow([num_rows as u64])
    }

    /// Precomputed sub-domain powers [1, ω_sub, ω_sub², ..., ω_sub^(num_rows-1)].
    pub(super) fn omega_powers(&self) -> Vec<F::BasePrimeField> {
        let num_rows = self.blinded_commitment.initial_committer.num_rows();
        crate::algebra::geometric_sequence(self.omega_sub(), num_rows)
    }

    /// Find the index of `alpha_base` in the sub-domain powers.
    pub(super) fn query_index(
        &self,
        alpha_base: F::BasePrimeField,
        omega_powers: &[F::BasePrimeField],
    ) -> usize {
        omega_powers
            .iter()
            .position(|&p| p == alpha_base)
            .expect("query point must be in IRS domain")
    }

    /// Compute the k coset gamma points for a query at `alpha_base`,
    /// lifted to the extension field via the blinded commitment embedding.
    pub(super) fn coset_gammas(
        &self,
        alpha_base: F::BasePrimeField,
        omega_powers: &[F::BasePrimeField],
    ) -> Vec<F> {
        use crate::algebra::embedding::Embedding;
        let embedding = self.blinded_commitment.embedding();
        let idx = self.query_index(alpha_base, omega_powers);
        let omega_full = self.omega_full();
        let zeta = self.zeta();
        let coset_offset = omega_full.pow([idx as u64]);
        let interleaving_depth = self.interleaving_depth();
        (0..interleaving_depth)
            .map(|coset_elem_idx| {
                let gamma_base = coset_offset * zeta.pow([coset_elem_idx as u64]);
                embedding.map(gamma_base)
            })
            .collect()
    }

    /// Compute all gamma points for a set of query points (flat list).
    pub(super) fn all_gammas(&self, query_points: &[F::BasePrimeField]) -> Vec<F> {
        let omega_powers = self.omega_powers();
        query_points
            .iter()
            .flat_map(|&alpha| self.coset_gammas(alpha, &omega_powers))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::{
        algebra::{
            fields::{Field64, Field64_2},
            linear_form::{Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;
    type EF = Field64_2;

    #[test]
    fn test_whir_zk_stage1_single_poly() {
        let num_variables = 8usize;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 16,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(2),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let params = Config::<EF>::new(mv_params, &whir_params, FoldingFactor::Constant(2), 1);

        let vector = vec![F::ONE; num_coeffs];
        let point = MultilinearPoint::rand(&mut rng, num_variables);
        let linear_form = MultilinearExtension {
            point: point.0.clone(),
        };
        let evaluation = linear_form.evaluate(params.blinded_commitment.embedding(), &vector);
        let linear_forms: Vec<Box<dyn LinearForm<EF>>> = vec![Box::new(linear_form)];
        let linear_form_refs = linear_forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1 {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &[&vector]);
        params.prove(
            &mut prover_state,
            &[&vector],
            &witness,
            &linear_form_refs,
            &[evaluation],
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params
            .receive_commitments(&mut verifier_state, 1)
            .expect("receive commitments");
        params
            .verify(
                &mut verifier_state,
                &linear_form_refs,
                &[evaluation],
                &commitment,
            )
            .expect("verify zk wrapper");
    }

    #[test]
    fn test_whir_zk_stage1_multi_poly_multi_query() {
        let num_variables = 8usize;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 16,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(2),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let params = Config::<EF>::new(mv_params, &whir_params, FoldingFactor::Constant(2), 2);

        let vector_0 = vec![F::ONE; num_coeffs];
        let vector_1 = (0..num_coeffs)
            .map(|i| F::from((i as u64).wrapping_mul(17).wrapping_add(3)))
            .collect::<Vec<_>>();
        let vectors = [&vector_0[..], &vector_1[..]];

        let point_0 = MultilinearPoint::rand(&mut rng, num_variables);
        let point_1 = MultilinearPoint::rand(&mut rng, num_variables);
        let linear_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let linear_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let eval_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let eval_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let linear_forms: Vec<Box<dyn LinearForm<EF>>> =
            vec![Box::new(linear_form_0), Box::new(linear_form_1)];
        let linear_form_refs = linear_forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();

        // Row-major evaluations: [w0(v0), w0(v1), w1(v0), w1(v1)].
        let mut evaluations = Vec::new();
        for lf in [&eval_form_0, &eval_form_1] {
            for vector in vectors {
                evaluations.push(lf.evaluate(params.blinded_commitment.embedding(), vector));
            }
        }

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1-multi {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &vectors);
        params.prove(
            &mut prover_state,
            &vectors,
            &witness,
            &linear_form_refs,
            &evaluations,
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params
            .receive_commitments(&mut verifier_state, 2)
            .expect("receive commitments");
        params
            .verify(
                &mut verifier_state,
                &linear_form_refs,
                &evaluations,
                &commitment,
            )
            .expect("verify zk wrapper multi");
    }

    #[test]
    fn test_whir_zk_stage1_rejects_wrong_evaluations() {
        let num_variables = 8usize;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 16,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(2),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let params = Config::<EF>::new(mv_params, &whir_params, FoldingFactor::Constant(2), 2);

        let vector_0 = vec![F::ONE; num_coeffs];
        let vector_1 = (0..num_coeffs)
            .map(|i| F::from((i as u64).wrapping_mul(29).wrapping_add(7)))
            .collect::<Vec<_>>();
        let vectors = [&vector_0[..], &vector_1[..]];

        let point_0 = MultilinearPoint::rand(&mut rng, num_variables);
        let point_1 = MultilinearPoint::rand(&mut rng, num_variables);
        let linear_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let linear_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let eval_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let eval_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let linear_forms: Vec<Box<dyn LinearForm<EF>>> =
            vec![Box::new(linear_form_0), Box::new(linear_form_1)];
        let linear_form_refs = linear_forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();

        let mut evaluations = Vec::new();
        for lf in [&eval_form_0, &eval_form_1] {
            for vector in vectors {
                evaluations.push(lf.evaluate(params.blinded_commitment.embedding(), vector));
            }
        }

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1-negative {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &vectors);
        params.prove(
            &mut prover_state,
            &vectors,
            &witness,
            &linear_form_refs,
            &evaluations,
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params
            .receive_commitments(&mut verifier_state, 2)
            .expect("receive commitments");

        let mut wrong_evaluations = evaluations.clone();
        wrong_evaluations[0] += EF::ONE;
        let verify_result = params.verify(
            &mut verifier_state,
            &linear_form_refs,
            &wrong_evaluations,
            &commitment,
        );
        assert!(
            verify_result.is_err(),
            "verification should reject wrong public evaluations"
        );
    }

    #[test]
    fn test_whir_zk_stage1_rejects_tampered_proof() {
        let num_variables = 8usize;
        let num_coeffs = 1 << num_variables;
        let mut rng = ark_std::test_rng();

        let mv_params = MultivariateParameters::new(num_variables);
        let whir_params = ProtocolParameters {
            initial_statement: true,
            security_level: 16,
            pow_bits: 0,
            folding_factor: FoldingFactor::Constant(2),
            soundness_type: SoundnessType::ConjectureList,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let params = Config::<EF>::new(mv_params, &whir_params, FoldingFactor::Constant(2), 2);

        let vector_0 = vec![F::ONE; num_coeffs];
        let vector_1 = (0..num_coeffs)
            .map(|i| F::from((i as u64).wrapping_mul(13).wrapping_add(11)))
            .collect::<Vec<_>>();
        let vectors = [&vector_0[..], &vector_1[..]];

        let point_0 = MultilinearPoint::rand(&mut rng, num_variables);
        let point_1 = MultilinearPoint::rand(&mut rng, num_variables);
        let linear_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let linear_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let eval_form_0 = MultilinearExtension {
            point: point_0.0.clone(),
        };
        let eval_form_1 = MultilinearExtension {
            point: point_1.0.clone(),
        };
        let linear_forms: Vec<Box<dyn LinearForm<EF>>> =
            vec![Box::new(linear_form_0), Box::new(linear_form_1)];
        let linear_form_refs = linear_forms
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<EF>)
            .collect::<Vec<_>>();

        let mut evaluations = Vec::new();
        for lf in [&eval_form_0, &eval_form_1] {
            for vector in vectors {
                evaluations.push(lf.evaluate(params.blinded_commitment.embedding(), vector));
            }
        }

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("zk-stage1-tamper {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = params.commit(&mut prover_state, &vectors);
        params.prove(
            &mut prover_state,
            &vectors,
            &witness,
            &linear_form_refs,
            &evaluations,
        );

        let mut proof = prover_state.proof();
        if let Some(last) = proof.narg_string.last_mut() {
            *last ^= 1;
        } else {
            panic!("expected non-empty proof transcript");
        }

        let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitment = params
                .receive_commitments(&mut verifier_state, 2)
                .expect("receive commitments");
            params.verify(
                &mut verifier_state,
                &linear_form_refs,
                &evaluations,
                &commitment,
            )
        }));
        match verify_outcome {
            Ok(result) => assert!(
                result.is_err(),
                "verification should reject tampered proof bytes"
            ),
            Err(_) => {
                // In debug transcript mode, tampering may trigger an interaction-pattern panic
                // before returning `Err`, which is still a valid rejection.
            }
        }
    }
}
