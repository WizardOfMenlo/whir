use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{BytesToUnitSerialize, FieldToUnitSerialize, UnitToField},
    ProofResult,
};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};

use super::{BatchingData, Witness};
#[cfg(debug_assertions)]
use crate::poly_utils::multilinear::MultilinearPoint;
use crate::{
    ntt::interleaved_rs_encode,
    poly_utils::coeffs::CoefficientList,
    whir::{
        parameters::WhirConfig,
        utils::{compute_ood_response, sample_ood_points, DigestToUnitSerialize},
    },
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
pub struct CommitmentWriter<F, MerkleConfig, PowStrategy>(
    pub(crate) WhirConfig<F, MerkleConfig, PowStrategy>,
)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    pub const fn new(config: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(config)
    }

    /// Commits a polynomial using a Merkle-based commitment scheme.
    ///
    /// This function:
    /// - Expands polynomial coefficients to evaluations.
    /// - Applies folding and restructuring optimizations.
    /// - Converts evaluations to an extension field.
    /// - Constructs a Merkle tree from the evaluations.
    /// - Returns a `Witness` containing the commitment data.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomial.num_coeffs())))]
    fn commit_single<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomial: &CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F> + UnitToField<F> + DigestToUnitSerialize<MerkleConfig>,
    {
        // Retrieve the base domain, ensuring it is set.
        let base_domain = self.0.starting_domain.base_domain.unwrap();

        // Compute expansion factor based on the domain size and polynomial length.
        let expansion = base_domain.size() / polynomial.num_coeffs();

        // Expand the polynomial coefficients into evaluations over the extended domain.
        let evals = interleaved_rs_encode(
            polynomial.coeffs(),
            expansion,
            self.0.folding_factor.at_round(0),
        );

        // Convert to extension field.
        // This is not necessary for the commit, but in further rounds
        // we will need the extension field. For symplicity we do it here too.
        // TODO: Commit to base field directly.
        let folded_evals = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "evals_to_extension", size = evals.len());
            evals
                .into_iter()
                .map(F::from_base_prime_field)
                .collect::<Vec<_>>()
        };

        // Determine leaf size based on folding factor.
        let fold_size = 1 << self.0.folding_factor.at_round(0);

        // Chunk evaluations into leaves for Merkle tree construction.
        #[cfg(not(feature = "parallel"))]
        let leafs_iter = folded_evals.chunks_exact(fold_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = folded_evals.par_chunks_exact(fold_size);

        // Construct the Merkle tree with given hash parameters.
        let merkle_tree = {
            #[cfg(feature = "tracing")]
            let _span = span!(Level::INFO, "MerkleTree::new", size = leafs_iter.len()).entered();
            MerkleTree::<MerkleConfig>::new(
                &self.0.leaf_hash_params,
                &self.0.two_to_one_params,
                leafs_iter,
            )
            .unwrap()
        };

        // Retrieve the Merkle tree root and add it to the narg_string.
        let root = merkle_tree.root();
        prover_state.add_digest(root)?;

        // Return the witness containing the polynomial, Merkle tree, and OOD results.
        Ok(Witness {
            polynomial: polynomial.clone().to_extension(),
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points: Vec::new(),
            ood_answers: Vec::new(),
            batching_data: Vec::new(),
            batching_randomness: F::zero(),
        })
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomials.first().unwrap().num_coeffs())))]
    pub fn commit_batch<ProverFSState>(
        &self,
        prover_state: &mut ProverFSState,
        polynomials: &[CoefficientList<F::BasePrimeField>],
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverFSState: FieldToUnitSerialize<F>
            + BytesToUnitSerialize
            + DigestToUnitSerialize<MerkleConfig>
            + UnitToField<F>,
    {
        assert!(!polynomials.is_empty());
        assert_eq!(polynomials.len(), self.0.batch_size);

        let num_vars = polynomials.first().unwrap().num_variables();
        let num_coeffs = polynomials.first().unwrap().num_coeffs();

        for poly in polynomials {
            assert_eq!(poly.num_variables(), num_vars);
            assert_eq!(poly.num_coeffs(), num_coeffs);
        }

        // 1. Create the Merkle tree and add _all_ the Merkle roots to transcript
        let mut witness_list = polynomials
            .into_iter()
            .map(|poly| self.commit_single(prover_state, poly))
            .collect::<ProofResult<Vec<_>>>()?;

        // 2. Randomly sample the OOD challenge and compute the OOD Response
        //    after _all_ merkle roots have been committed. Returns the
        //    challenge OOD point that was used for computing the responses.
        let ood_points = witness_list
            .iter_mut()
            .try_fold(
                None as Option<Vec<F>>, /* OOD challenge point */
                |ood_points, witness| {
                    if let Some(ood_points) = ood_points {
                        // OOD challenge points were previously computed
                        compute_ood_response(
                            prover_state,
                            &ood_points,
                            self.0.mv_parameters.num_variables,
                            |point| witness.polynomial.evaluate(point),
                        )
                        .map(|ood_response| {
                            witness.ood_answers = ood_response;
                            Some(ood_points)
                        })
                    } else {
                        // Compute the challenge OOD points and OOD response
                        sample_ood_points(
                            prover_state,
                            self.0.committment_ood_samples,
                            self.0.mv_parameters.num_variables,
                            |point| witness.polynomial.evaluate(point),
                        )
                        .map(|(ood_point, ood_response)| {
                            witness.ood_answers = ood_response;
                            Some(ood_point)
                        })
                    }
                },
            )?
            .unwrap_or_default();

        if witness_list.len() == 1 {
            // Without batching no extra step is needed
            return Ok(Witness {
                ood_points,
                ..witness_list.into_iter().next().unwrap()
            });
        }

        // Sample batching_randomness
        let [batching_randomness] = prover_state.challenge_scalars()?;

        Ok(Witness::new_batched(
            witness_list,
            ood_points,
            batching_randomness,
            &self.0,
        ))
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(size = polynomial.num_coeffs())))]
    pub fn commit<ProverState>(
        &self,
        prover_state: &mut ProverState,
        polynomial: CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F>
            + UnitToField<F>
            + DigestToUnitSerialize<MerkleConfig>
            + BytesToUnitSerialize,
    {
        self.commit_batch(prover_state, &[polynomial])
    }
}

impl<F, MerkleConfig> Witness<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
{
    // Takes a list of witnesses and computes the top level batching witness.
    fn new_batched<PowStrategy>(
        witness_list: Vec<Witness<F, MerkleConfig>>,
        ood_points: Vec<F>,
        batching_randomness: F,
        config: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        assert!(
            witness_list.len() > 1,
            "Witnesses of size 1 should be handled elsewhere"
        );

        let num_vars = witness_list.first().unwrap().polynomial.num_variables();

        for wit in &witness_list {
            assert!(wit.polynomial.coeffs().len().is_power_of_two());
            assert!(wit.merkle_leaves.len().is_power_of_two());
            assert_eq!(wit.polynomial.num_variables(), num_vars);
            assert_eq!(wit.merkle_leaves.len(), witness_list[0].merkle_leaves.len())
        }

        let poly_dim = witness_list.first().unwrap().polynomial.coeffs().len();
        let oracle_len = witness_list.first().unwrap().merkle_leaves.len();

        assert_eq!(oracle_len % poly_dim, 0);

        let rate_factor = oracle_len / poly_dim;

        let (first_witness, rest_witnesses) = witness_list.split_first().unwrap();

        let mut batched_poly: Vec<F> = first_witness.polynomial.coeffs().to_vec();
        let mut batched_oracle: Vec<F> = first_witness.merkle_leaves.clone();
        let mut batched_ood_resp: Vec<F> = first_witness.ood_answers.clone();
        let mut multiplier = batching_randomness;

        for wit in rest_witnesses {
            // TODO: Get rid of index based updates.
            for (i, coefficient) in wit.polynomial.coeffs().iter().enumerate() {
                batched_poly[i] += multiplier * coefficient;
                for j in 0..rate_factor {
                    let ndx = i + j * poly_dim;
                    batched_oracle[ndx] += multiplier * wit.merkle_leaves[ndx];
                }
            }

            batched_ood_resp
                .iter_mut()
                .zip(wit.ood_answers.iter())
                .for_each(|(ood_resp, val)| *ood_resp += *val * multiplier);

            multiplier *= batching_randomness;
        }

        let polynomial = CoefficientList::new(batched_poly);

        #[cfg(debug_assertions)]
        {
            for (ood_point, ood_resp) in ood_points.iter().zip(batched_ood_resp.iter()) {
                let expected = polynomial.evaluate(&MultilinearPoint::expand_from_univariate(
                    ood_point.clone(),
                    num_vars,
                ));
                assert_eq!(expected, ood_resp.clone());
            }
        }

        let batched_tree = MerkleTree::<MerkleConfig>::blank(
            &config.leaf_hash_params,
            &config.two_to_one_params,
            (ark_std::log2(batched_oracle.len()) as usize) + 1,
        )
        .unwrap();

        Self {
            polynomial,
            merkle_leaves: batched_oracle,
            merkle_tree: batched_tree,
            ood_points,
            ood_answers: batched_ood_resp,
            batching_randomness,
            batching_data: witness_list.into_iter().map(BatchingData::from).collect(),
        }
    }

    pub fn roots(&self) -> Vec<MerkleConfig::InnerDigest> {
        if self.batching_data.is_empty() {
            vec![self.merkle_tree.root()]
        } else {
            self.batching_data
                .iter()
                .map(|w| w.merkle_tree.root())
                .collect()
        }
    }

    /// Returns the batched polynomial
    pub fn batched_poly(&self) -> &CoefficientList<F> {
        &self.polynomial
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::UniformRand;
    use spongefish::DomainSeparator;
    use spongefish_pow::blake3::Blake3PoW;

    use super::*;
    use crate::{
        crypto::{
            fields::Field64,
            merkle_tree::{
                keccak::{KeccakCompress, KeccakLeafHash, KeccakMerkleTreeParams},
                parameters::default_config,
            },
        },
        parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
        poly_utils::multilinear::MultilinearPoint,
        whir::domainsep::WhirDomainSeparator,
    };

    #[test]
    fn test_basic_commitment() {
        // Define the field type and Merkle tree configuration.
        type F = Field64;
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();

        // Generate Merkle tree hash parameters
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let whir_params = ProtocolParameters::<MerkleConfig, Blake3PoW> {
            initial_statement: true,
            security_level,
            pow_bits,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            leaf_hash_params,
            two_to_one_params,
            soundness_type: SoundnessType::ConjectureList,
            _pow_parameters: std::marker::PhantomData,
            starting_log_inv_rate: starting_rate,
            batch_size: 1,
        };

        // Define multivariate parameters for the polynomial.
        let mv_params = MultivariateParameters::<F>::new(num_variables);
        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(mv_params, whir_params);

        // Generate a random polynomial with 32 coefficients.
        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let domainsep = DomainSeparator::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params);
        let mut prover_state = domainsep.to_prover_state();

        // Run the Commitment Phase
        let committer = CommitmentWriter::new(params.clone());
        let witness = committer
            .commit(&mut prover_state, polynomial.clone())
            .unwrap();

        // Ensure Merkle leaves are correctly generated.
        assert!(
            !witness.merkle_leaves.is_empty(),
            "Merkle leaves should not be empty"
        );

        // Ensure OOD (out-of-domain) points are generated.
        assert!(
            !witness.ood_points.is_empty(),
            "OOD points should be generated"
        );

        // Validate the number of generated OOD points.
        assert_eq!(
            witness.ood_points.len(),
            params.committment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that the Merkle tree root is valid
        let root = witness.merkle_tree.root();
        assert_ne!(
            root.as_ref(),
            &[0u8; 32],
            "Merkle tree root should not be zero"
        );

        // Ensure polynomial data is correctly stored
        assert_eq!(
            witness.polynomial.coeffs().len(),
            polynomial.coeffs().len(),
            "Stored polynomial should have the correct number of coefficients"
        );

        // Check that OOD answers match expected evaluations
        for (i, ood_point) in witness.ood_points.iter().enumerate() {
            let expected_eval = polynomial.evaluate_at_extension(
                &MultilinearPoint::expand_from_univariate(*ood_point, num_variables),
            );
            assert_eq!(
                witness.ood_answers[i], expected_eval,
                "OOD answer at index {i} should match expected evaluation"
            );
        }
    }

    #[test]
    fn test_large_polynomial() {
        type F = Field64;
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        let params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(10),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
            },
        );

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 1024]); // Large polynomial
        let domainsep = DomainSeparator::new("🌪️").commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, polynomial).unwrap();

        // Expansion factor is 2
        assert_eq!(
            witness.merkle_leaves.len(),
            1024 * 2,
            "Merkle tree should have expected number of leaves"
        );
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        type F = Field64;
        type MerkleConfig = KeccakMerkleTreeParams<F>;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);

        let mut params = WhirConfig::<F, MerkleConfig, Blake3PoW>::new(
            MultivariateParameters::<F>::new(5),
            ProtocolParameters {
                initial_statement: true,
                security_level: 100,
                pow_bits: 20,
                folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
                leaf_hash_params,
                two_to_one_params,
                soundness_type: SoundnessType::ConjectureList,
                _pow_parameters: Default::default(),
                starting_log_inv_rate: 1,
                batch_size: 1,
            },
        );

        params.committment_ood_samples = 0; // No OOD samples

        let polynomial = CoefficientList::new(vec![F::rand(&mut rng); 32]);
        let domainsep = DomainSeparator::new("🌪️").commit_statement(&params);
        let mut prover_state = domainsep.to_prover_state();

        let committer = CommitmentWriter::new(params);
        let witness = committer.commit(&mut prover_state, polynomial).unwrap();

        assert!(
            witness.ood_points.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
