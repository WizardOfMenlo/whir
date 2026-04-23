/// Soundness regression tests for evaluation forgery attacks.
///
/// All three issues share the same root cause: public evaluations were not
/// bound in the Fiat-Shamir transcript before the challenges that depend on
/// them (α, ρ, constraint_rlc).
///
/// - **Issue #1 (α):** n > 1 polynomials batched via `Σ αⁱ·eᵢ`.
///   Forgery: `[+Δ, −Δ/α]` preserves the weighted sum.
/// - **Issue #2 (ρ):** zkWHIR combines `claim = ρ·e + G`.
///   Forgery: send `G' = G+Δ`, then `e' = e−Δ/ρ`.
/// - **Issue #3 (constraint_rlc):** f > 1 forms collapsed via `Σ cⱼ·claimⱼ`.
///   Forgery: `[+Δ, −Δ/c₁]` preserves the weighted sum.
///
/// **Fix:** absorb all evaluations into the transcript before α, ρ, and
/// constraint_rlc are sampled.  Verifier reads them back and checks
/// `verify!(read == expected)`.

// =========================================================================
// WHIR (non-ZK)
// =========================================================================

#[cfg(test)]
mod whir_tests {
    use std::borrow::Cow;

    use ark_ff::Field;

    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64, Field64_3},
            linear_form::{Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        protocols::{geometric_challenge::geometric_challenge, whir},
        transcript::{codecs::Empty, DomainSeparator, Proof, ProverState, VerifierState},
    };

    type F = Field64;
    type EF = Field64_3;

    const NUM_VARS: usize = 4;
    const NUM_COEFFS: usize = 1 << NUM_VARS;

    fn make_config(batch_size: usize) -> whir::Config<Basefield<EF>> {
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            unique_decoding: false,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };
        let mut config = whir::Config::<Basefield<EF>>::new(NUM_COEFFS, &params);
        config.disable_pow();
        config
    }

    fn make_forms(points: &[MultilinearPoint<EF>]) -> Vec<Box<dyn Evaluate<Basefield<EF>>>> {
        points
            .iter()
            .map(|p| Box::new(MultilinearExtension { point: p.0.clone() }) as _)
            .collect()
    }

    fn prove_forms(points: &[MultilinearPoint<EF>]) -> Vec<Box<dyn LinearForm<EF>>> {
        points
            .iter()
            .map(|p| {
                Box::new(MultilinearExtension { point: p.0.clone() }) as Box<dyn LinearForm<EF>>
            })
            .collect()
    }

    /// Attempt verification with `claimed_evals`.
    /// Returns true if accepted (soundness bug), false if correctly rejected.
    fn forgery_accepted_separate(
        config: &whir::Config<Basefield<EF>>,
        ds: &DomainSeparator<'_, Empty>,
        proof: &Proof,
        forms: &[Box<dyn Evaluate<Basefield<EF>>>],
        claimed_evals: &[EF],
        num_commits: usize,
    ) -> bool {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(ds, proof);
            let commitments: Vec<_> = (0..num_commits)
                .map(|_| config.receive_commitment(&mut vs).unwrap())
                .collect();
            let refs: Vec<_> = commitments.iter().collect();
            config
                .verify(&mut vs, &refs, claimed_evals)
                .and_then(|fc| fc.verify(forms.iter().map(|l| l.as_ref() as &dyn LinearForm<EF>)))
        }));
        matches!(result, Ok(Ok(())))
    }

    // ─── Issue #1: α-batching forgery ────────────────────────────────────

    /// Issue #1, separate commitments (batch_size=1, n=2, f=1).
    #[test]
    fn test_whir_issue1_separate_commits() {
        let config = make_config(1);
        let mut rng = ark_std::test_rng();

        let v0 = vec![F::ONE; NUM_COEFFS];
        let v1 = vec![F::from(2u64); NUM_COEFFS];
        let points = vec![MultilinearPoint::rand(&mut rng, NUM_VARS)];
        let forms = make_forms(&points);
        let evals: Vec<EF> = forms
            .iter()
            .flat_map(|lf| [&v0, &v1].map(|v| lf.evaluate(config.embedding(), v)))
            .collect();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let w0 = config.commit(&mut ps, &[&v0]);
        let w1 = config.commit(&mut ps, &[&v1]);
        let _ = config.prove(
            &mut ps,
            vec![Cow::Borrowed(&v0[..]), Cow::Borrowed(&v1[..])],
            vec![Cow::Owned(w0), Cow::Owned(w1)],
            prove_forms(&points),
            Cow::Borrowed(&evals),
        );
        let proof = ps.proof();

        // Sanity.
        assert!(forgery_accepted_separate(
            &config, &ds, &proof, &forms, &evals, 2
        ));

        // Forge one entry.
        let mut forged = evals.clone();
        forged[0] += EF::from(1u64);
        assert!(
            !forgery_accepted_separate(&config, &ds, &proof, &forms, &forged, 2),
            "REGRESSION issue #1: single-entry forgery (separate commits) must be rejected"
        );
    }

    /// Issue #1, batched commitment (batch_size=2, n=2, f=1).
    #[test]
    fn test_whir_issue1_batched_commit() {
        let config = make_config(2);
        let mut rng = ark_std::test_rng();

        let v0 = vec![F::ONE; NUM_COEFFS];
        let v1 = vec![F::from(3u64); NUM_COEFFS];
        let vec_refs: Vec<&[F]> = vec![&v0[..], &v1[..]];
        let points = vec![MultilinearPoint::rand(&mut rng, NUM_VARS)];
        let forms = make_forms(&points);
        let evals: Vec<EF> = forms
            .iter()
            .flat_map(|lf| vec_refs.iter().map(|v| lf.evaluate(config.embedding(), v)))
            .collect();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let w = config.commit(&mut ps, &vec_refs);
        let _ = config.prove(
            &mut ps,
            vec![Cow::Borrowed(&v0[..]), Cow::Borrowed(&v1[..])],
            vec![Cow::Owned(w)],
            prove_forms(&points),
            Cow::Borrowed(&evals),
        );
        let proof = ps.proof();

        assert!(forgery_accepted_separate(
            &config, &ds, &proof, &forms, &evals, 1
        ));

        let mut forged = evals.clone();
        forged[0] += EF::from(1u64);
        assert!(
            !forgery_accepted_separate(&config, &ds, &proof, &forms, &forged, 1),
            "REGRESSION issue #1: single-entry forgery (batched commit) must be rejected"
        );
    }

    /// Issue #1, exact α-cancelling forgery (n=2, f=1).
    /// Extracts α from transcript, constructs `[+Δ, −Δ/α]`.
    #[test]
    fn test_whir_issue1_alpha_cancelling() {
        let config = make_config(1);
        let mut rng = ark_std::test_rng();

        let v0 = vec![F::ONE; NUM_COEFFS];
        let v1 = vec![F::from(2u64); NUM_COEFFS];
        let points = vec![MultilinearPoint::rand(&mut rng, NUM_VARS)];
        let forms = make_forms(&points);
        let evals: Vec<EF> = forms
            .iter()
            .flat_map(|lf| [&v0, &v1].map(|v| lf.evaluate(config.embedding(), v)))
            .collect();
        assert_eq!(evals.len(), 2);

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let w0 = config.commit(&mut ps, &[&v0]);
        let w1 = config.commit(&mut ps, &[&v1]);
        let _ = config.prove(
            &mut ps,
            vec![Cow::Borrowed(&v0[..]), Cow::Borrowed(&v1[..])],
            vec![Cow::Owned(w0), Cow::Owned(w1)],
            prove_forms(&points),
            Cow::Borrowed(&evals),
        );
        let proof = ps.proof();

        // Extract α by replaying the verifier transcript prefix.
        // After receive_commitment, verify() reads OOD cross-terms,
        // evaluations, then squeezes vector_rlc_coeffs = [1, α].
        let alpha = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let c0 = config.receive_commitment(&mut vs).unwrap();
            let c1 = config.receive_commitment(&mut vs).unwrap();
            // OOD cross-terms: 1 per commitment per OOD row.
            for _ in 0..(c0.out_of_domain().points.len() + c1.out_of_domain().points.len()) {
                let _: EF = vs.prover_message().unwrap();
            }
            for _ in 0..evals.len() {
                let _: EF = vs.prover_message().unwrap();
            }
            let coeffs: Vec<EF> = geometric_challenge(&mut vs, 2);
            coeffs[1]
        };

        // Exact cancelling forgery: e'₀ + α·e'₁ = e₀ + α·e₁.
        let delta = EF::from(42u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / alpha;
        assert_eq!(evals[0] + alpha * evals[1], forged[0] + alpha * forged[1]);

        assert!(
            !forgery_accepted_separate(&config, &ds, &proof, &forms, &forged, 2),
            "REGRESSION issue #1: α-cancelling forgery [+Δ, −Δ/α] must be rejected"
        );
    }

    // ─── Issue #3: constraint-RLC forgery ────────────────────────────────

    /// Issue #3, exact c₁-cancelling forgery (n=1, f=2).
    /// Extracts c₁ from transcript, constructs `[+Δ, −Δ/c₁]`.
    #[test]
    fn test_whir_issue3_constraint_rlc_cancelling() {
        let config = make_config(1);
        let mut rng = ark_std::test_rng();

        let vector = vec![F::ONE; NUM_COEFFS];
        let points: Vec<_> = (0..2)
            .map(|_| MultilinearPoint::rand(&mut rng, NUM_VARS))
            .collect();
        let forms = make_forms(&points);
        let evals: Vec<EF> = forms
            .iter()
            .flat_map(|lf| [&vector].map(|v| lf.evaluate(config.embedding(), v)))
            .collect();
        assert_eq!(evals.len(), 2);

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let w = config.commit(&mut ps, &[&vector]);
        let _ = config.prove(
            &mut ps,
            vec![Cow::Borrowed(vector.as_slice())],
            vec![Cow::Owned(w)],
            prove_forms(&points),
            Cow::Borrowed(&evals),
        );
        let proof = ps.proof();

        // Extract c₁.  After commitment + evaluations, the verifier squeezes
        // vector_rlc (count=1, no squeeze) then constraint_rlc (count = ood + 2).
        let c1 = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let c = config.receive_commitment(&mut vs).unwrap();
            for _ in 0..evals.len() {
                let _: EF = vs.prover_message().unwrap();
            }
            let _: Vec<EF> = geometric_challenge(&mut vs, 1); // vector_rlc [1]
            let num_ood = c.out_of_domain().points.len();
            let rlc: Vec<EF> = geometric_challenge(&mut vs, num_ood + 2);
            rlc[1]
        };

        // Exact cancelling forgery: e'₀ + c₁·e'₁ = e₀ + c₁·e₁.
        let delta = EF::from(99u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / c1;
        assert_eq!(evals[0] + c1 * evals[1], forged[0] + c1 * forged[1]);

        assert!(
            !forgery_accepted_separate(&config, &ds, &proof, &forms, &forged, 1),
            "REGRESSION issue #3: constraint-RLC-cancelling forgery must be rejected"
        );
    }
}

// =========================================================================
// zkWHIR
// =========================================================================

#[cfg(test)]
mod whir_zk_tests {
    use std::borrow::Cow;

    use ark_ff::{AdditiveGroup, Field};

    use crate::{
        algebra::{
            dot,
            embedding::Identity,
            fields::Field64,
            geometric_sequence,
            linear_form::{
                Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation,
            },
            multilinear_extend, univariate_evaluate, MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        protocols::{
            geometric_challenge::geometric_challenge,
            whir,
            whir_zk::{
                self,
                committer::Witness,
                utils::{
                    build_beq_tables, build_fold_args, build_weight_covectors, compute_eq_weights,
                    compute_rs_fold_blinding_coeffs, gamma_to_f_hat_indices, ProtocolDims,
                    RsFoldCoeffs,
                },
            },
        },
        transcript::{
            codecs::Empty, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
        },
    };

    type F = Field64;

    const NUM_VARS: usize = 12;
    const NUM_COEFFS: usize = 1 << NUM_VARS;

    fn make_config(batch_size: usize) -> whir_zk::Config<F> {
        let params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };
        let mut config = whir_zk::Config::new(NUM_VARS, &params);
        config.disable_pow();
        config
    }

    fn to_prove_forms(
        forms: &[Box<dyn LinearForm<F>>],
        size: usize,
    ) -> Vec<Box<dyn LinearForm<F>>> {
        forms
            .iter()
            .map(|f| {
                let mut cv = vec![F::ZERO; size];
                f.accumulate(&mut cv, F::ONE);
                Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>
            })
            .collect()
    }

    /// Generate an honest proof and sanity-check it.
    fn honest_proof(
        config: &whir_zk::Config<F>,
        vectors: &[&[F]],
        forms: &[Box<dyn LinearForm<F>>],
        evals: &[F],
    ) -> (DomainSeparator<'static, Empty>, Proof) {
        let ds = DomainSeparator::protocol(config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let witness = config.commit(&mut ps, vectors);
        config.prove(
            &mut ps,
            vectors.iter().map(|&v| Cow::Borrowed(v)).collect(),
            witness,
            to_prove_forms(forms, vectors[0].len()),
            Cow::Borrowed(evals),
        );
        let proof = ps.proof();

        // Sanity: honest proof must pass.
        let weights: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = config.receive_commitments(&mut vs).unwrap();
        config
            .verify(&mut vs, &weights, evals, &commitments)
            .unwrap()
            .verify(weights.iter().copied())
            .unwrap();

        (ds, proof)
    }

    /// Attempt verification with `claimed_evals`.  Returns true on accept.
    fn forgery_accepted(
        config: &whir_zk::Config<F>,
        ds: &DomainSeparator<'_, Empty>,
        proof: &Proof,
        forms: &[Box<dyn LinearForm<F>>],
        claimed_evals: &[F],
    ) -> bool {
        let weights: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(ds, proof);
            let commitments = config.receive_commitments(&mut vs).unwrap();
            config
                .verify(&mut vs, &weights, claimed_evals, &commitments)
                .and_then(|fc| fc.verify(weights.iter().copied()))
        }));
        matches!(result, Ok(Ok(())))
    }

    // ─── Issue #1: α-batching forgery ────────────────────────────────────

    /// Issue #1 (n=2, f=1): exact α-cancelling forgery.
    /// Extracts α from transcript, constructs `[+Δ, −Δ/α]`.
    #[test]
    fn test_zkwhir_issue1_alpha_cancelling() {
        let config = make_config(2);
        let mut rng = ark_std::test_rng();

        let v0: Vec<F> = (0..NUM_COEFFS).map(|i| F::from(i as u64 + 1)).collect();
        let v1: Vec<F> = (0..NUM_COEFFS).map(|i| F::from(i as u64 * 3 + 7)).collect();
        let point = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let form = MultilinearExtension { point: point.0 };
        let embedding = config.embedding();
        let evals = vec![form.evaluate(embedding, &v0), form.evaluate(embedding, &v1)];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];

        let (ds, proof) = honest_proof(&config, &[&v0, &v1], &forms, &evals);

        // Extract α.  After receive_commitments: squeeze β, 1 g_claim, 2 evals, squeeze α.
        let alpha = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let _ = config.receive_commitments(&mut vs).unwrap();
            let _beta: F = vs.verifier_message();
            let _g: F = vs.prover_message().unwrap();
            let _e0: F = vs.prover_message().unwrap();
            let _e1: F = vs.prover_message().unwrap();
            geometric_challenge::<_, F>(&mut vs, 2)[1]
        };

        let delta = F::from(42u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / alpha;
        assert_eq!(evals[0] + alpha * evals[1], forged[0] + alpha * forged[1]);

        assert!(
            !forgery_accepted(&config, &ds, &proof, &forms, &forged),
            "REGRESSION issue #1: α-cancelling forgery must be rejected"
        );
    }

    // ─── Issue #2: G-claim forgery via ρ ─────────────────────────────────

    /// Issue #2 (n=1, f=1): full manual transcript replay with forged g_claim.
    ///
    /// 1. Commit honestly.
    /// 2. Send G' = G + Δ.
    /// 3. Absorb honest eval (must commit before ρ).
    /// 4. ρ is squeezed → construct e' = e − Δ/ρ.
    /// 5. Complete proof honestly.
    /// 6. Verifier reads honest e from transcript, compares to e' → rejected.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_zkwhir_issue2_g_claim_forgery() {
        let mut rng = ark_std::test_rng();
        let config = make_config(1);

        let vector = vec![F::ONE; NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let form = MultilinearExtension { point: point.0 };
        let honest_eval = form.evaluate(config.embedding(), &vector);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];
        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &[&vector]);

        let Witness {
            f_hat_witness,
            blinding_poly_witness,
            f_hat_polys,
            secrets,
        } = witness;

        let dims = ProtocolDims::new(&config, 1);
        let size = dims.size;

        // ── MALICIOUS Step 2: send g_claim + Δ ──

        let beta: F = prover_state.verifier_message();
        let beta_powers = geometric_sequence(beta, dims.num_g_polys());
        let g_poly: Vec<F> = (0..size)
            .map(|idx| {
                beta_powers
                    .iter()
                    .enumerate()
                    .map(|(i, &bp)| bp * secrets.g_polys[i][dims.phi_i_bits(idx, i)])
                    .sum()
            })
            .collect();
        let honest_g_claim: F = {
            let mut buf = vec![F::ZERO; size];
            forms[0].accumulate(&mut buf, F::ONE);
            dot(&buf, &g_poly)
        };

        let delta = F::from(77u64);
        prover_state.prover_message(&(honest_g_claim + delta)); // forged G'
        prover_state.prover_message(&honest_eval); // must absorb before ρ

        let alpha_coeffs: Vec<F> = geometric_challenge(&mut prover_state, 1);
        let rho: F = prover_state.verifier_message();
        assert_ne!(rho, F::ZERO);

        // f_zk = ρ·f + g (honest)
        let mut f_zk: Vec<F> = vector.iter().map(|&v| rho * v).collect();
        for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
            *f += g;
        }
        drop(g_poly);

        let combined_claim = rho * honest_eval + honest_g_claim;

        // Step 4: sumcheck
        let constraint_rlc: Vec<F> = geometric_challenge(&mut prover_state, 1);
        let mut covector = vec![F::ZERO; size];
        for (coeff, lf) in constraint_rlc.iter().zip(forms.iter()) {
            lf.accumulate(&mut covector, *coeff);
        }
        let mut the_sum: F = constraint_rlc[0] * combined_claim;
        let folding_randomness = config.blinded_polynomial.initial_sumcheck.prove(
            &mut prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        // Steps 5-6: honest
        let r_bar = &folding_randomness.0;
        let eq_weights_vec = compute_eq_weights(r_bar);
        let RsFoldCoeffs {
            masking_coeffs_all,
            g_i_coeffs,
        } = compute_rs_fold_blinding_coeffs(
            &eq_weights_vec,
            &secrets.g_polys,
            &secrets.masking_polys,
            &alpha_coeffs,
            rho,
            dims,
        );

        let round_config = &config.blinded_polynomial.round_configs[0];
        let folded_commit = round_config
            .irs_committer
            .commit(&mut prover_state, &[&f_zk]);
        round_config.pow.prove(&mut prover_state);
        let in_domain = config
            .blinded_polynomial
            .initial_committer
            .open(&mut prover_state, &[&f_hat_witness]);

        let mut lambda_z_points: Vec<F> = Vec::new();
        let send_blinding = |ps: &mut ProverState<_, _>, z: F| {
            for m in &masking_coeffs_all {
                ps.prover_message(&univariate_evaluate(m, z));
            }
            for g in &g_i_coeffs {
                ps.prover_message(&univariate_evaluate(g, z));
            }
        };

        let f_hat_combined = &f_hat_polys[0];
        let mu = dims.mu;
        for &z in &folded_commit.out_of_domain().points {
            prover_state.prover_message(&multilinear_extend(
                f_hat_combined,
                &build_fold_args(r_bar, z, mu),
            ));
            send_blinding(&mut prover_state, z);
            lambda_z_points.push(z);
        }
        drop(f_hat_polys);
        for &z in &in_domain.points {
            send_blinding(&mut prover_state, z);
            lambda_z_points.push(z);
        }
        {
            let stir_challenges: Vec<UnivariateEvaluation<F>> = folded_commit
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect();
            let ood_evals = folded_commit.out_of_domain().values(&[F::ONE]);
            let num_ood = folded_commit.out_of_domain().points.len();
            let embedding = Identity::new();
            let stir_evals: Vec<F> = ood_evals
                .chain(
                    stir_challenges[num_ood..]
                        .iter()
                        .map(|ch| ch.evaluate(&embedding, &f_zk)),
                )
                .collect();
            let stir_rlc: Vec<F> = geometric_challenge(&mut prover_state, stir_challenges.len());
            UnivariateEvaluation::accumulate_many(&stir_challenges, &mut covector, &stir_rlc);
            the_sum += dot(&stir_rlc, &stir_evals);
        }
        let round0_folding =
            round_config
                .sumcheck
                .prove(&mut prover_state, &mut f_zk, &mut covector, &mut the_sum);
        let remaining = whir::rounds::prove_remaining_rounds(
            &config.blinded_polynomial.round_configs,
            &whir::rounds::FinalRoundConfig {
                sumcheck: &config.blinded_polynomial.final_sumcheck,
                pow: &config.blinded_polynomial.final_pow,
            },
            &mut prover_state,
            &mut whir::rounds::SumcheckState {
                vector: &mut f_zk,
                covector: &mut covector,
                the_sum: &mut the_sum,
            },
            folded_commit,
            &round0_folding,
        );
        let gamma_points = remaining.first_in_domain_points;
        let _ = config.blinded_polynomial.initial_committer.open_at_indices(
            &mut prover_state,
            &[&f_hat_witness],
            &gamma_to_f_hat_indices(&gamma_points, &config),
        );
        for &gamma in &gamma_points {
            send_blinding(&mut prover_state, gamma);
            lambda_z_points.push(gamma);
        }
        drop(f_zk);
        drop(covector);

        // Step 7: blinding proof (honest)
        let tau: F = prover_state.verifier_message();
        let beq_tables = build_beq_tables(&lambda_z_points, &eq_weights_vec, tau, dims);
        let weight_covectors = build_weight_covectors(&beq_tables, rho, &alpha_coeffs, dims);
        let mut eval_matrix = Vec::with_capacity(dims.num_blinding_vecs * dims.num_blinding_vecs);
        for w in &weight_covectors {
            for v in &secrets.blinding_vectors {
                eval_matrix.push(dot(w, v));
            }
        }
        for e in &eval_matrix {
            prover_state.prover_message(e);
        }
        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as _)
            .collect();
        let blinding_cows: Vec<Cow<'_, [F]>> = secrets
            .blinding_vectors
            .iter()
            .map(|v| Cow::Borrowed(v.as_slice()))
            .collect();
        let _ = config.blinding_polynomial.prove(
            &mut prover_state,
            blinding_cows,
            vec![Cow::Borrowed(&blinding_poly_witness)],
            blinding_forms,
            Cow::Owned(eval_matrix),
        );

        // Verify with forged e' = e − Δ/ρ.
        let proof = prover_state.proof();
        let forged_eval = honest_eval - delta / rho;
        assert_ne!(forged_eval, honest_eval);

        let attack = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitments = config.receive_commitments(&mut vs).unwrap();
            config
                .verify(&mut vs, &weight_refs, &[forged_eval], &commitments)
                .and_then(|fc| fc.verify(weight_refs.iter().copied()))
        }));
        assert!(
            !matches!(attack, Ok(Ok(()))),
            "REGRESSION issue #2: g_claim forgery (G'=G+Δ, e'=e−Δ/ρ) must be rejected"
        );
    }

    // ─── Issue #3: constraint-RLC forgery ────────────────────────────────

    /// Issue #3 (n=1, f=2): exact c₁-cancelling forgery.
    /// Extracts c₁ from transcript, constructs `[+Δ, −Δ/c₁]`.
    #[test]
    fn test_zkwhir_issue3_constraint_rlc_cancelling() {
        let config = make_config(1);
        let mut rng = ark_std::test_rng();

        let vector: Vec<F> = (0..NUM_COEFFS).map(|i| F::from(i as u64 + 1)).collect();
        let p0 = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let p1 = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let embedding = config.embedding();
        let evals = vec![
            f0.evaluate(embedding, &vector),
            f1.evaluate(embedding, &vector),
        ];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];

        let (ds, proof) = honest_proof(&config, &[&vector], &forms, &evals);

        // Extract c₁.  For n=1, f=2:
        //   β, 2 g_claims, 2 evals, α(1→no squeeze), ρ, constraint_rlc=[1, c₁].
        let c1 = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let _ = config.receive_commitments(&mut vs).unwrap();
            let _beta: F = vs.verifier_message();
            for _ in 0..2 {
                let _: F = vs.prover_message().unwrap();
            } // g_claims
            for _ in 0..2 {
                let _: F = vs.prover_message().unwrap();
            } // evals
            let _: Vec<F> = geometric_challenge(&mut vs, 1); // α = [1]
            let _rho: F = vs.verifier_message();
            geometric_challenge::<_, F>(&mut vs, 2)[1]
        };

        let delta = F::from(99u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / c1;
        assert_eq!(evals[0] + c1 * evals[1], forged[0] + c1 * forged[1]);

        assert!(
            !forgery_accepted(&config, &ds, &proof, &forms, &forged),
            "REGRESSION issue #3: constraint-RLC-cancelling forgery must be rejected"
        );
    }

    // ─── Combined: all three surfaces ────────────────────────────────────

    /// Issues #1+#2+#3 combined (n=2, f=2): 4 evaluations.
    /// Tests single-entry, cross-vector, and cross-form forgeries.
    #[test]
    fn test_zkwhir_combined_n2_f2() {
        let config = make_config(2);
        let mut rng = ark_std::test_rng();

        let v0: Vec<F> = (0..NUM_COEFFS).map(|i| F::from(i as u64 + 1)).collect();
        let v1: Vec<F> = (0..NUM_COEFFS).map(|i| F::from(i as u64 * 3 + 7)).collect();
        let p0 = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let p1 = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let embedding = config.embedding();
        let evals = vec![
            f0.evaluate(embedding, &v0),
            f0.evaluate(embedding, &v1),
            f1.evaluate(embedding, &v0),
            f1.evaluate(embedding, &v1),
        ];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];

        let (ds, proof) = honest_proof(&config, &[&v0, &v1], &forms, &evals);

        // (a) Single entry.
        let mut fa = evals.clone();
        fa[0] += F::from(1u64);
        assert!(
            !forgery_accepted(&config, &ds, &proof, &forms, &fa),
            "single-entry forgery must be rejected"
        );

        // (b) Cross-vector (α dimension, row 0).
        let mut fb = evals.clone();
        fb[0] += F::from(99u64);
        fb[1] -= F::from(99u64);
        assert!(
            !forgery_accepted(&config, &ds, &proof, &forms, &fb),
            "cross-vector forgery must be rejected"
        );

        // (c) Cross-form (constraint-RLC dimension).
        let mut fc = evals.clone();
        fc[0] += F::from(55u64);
        fc[2] -= F::from(55u64);
        assert!(
            !forgery_accepted(&config, &ds, &proof, &forms, &fc),
            "cross-form forgery must be rejected"
        );
    }
}
