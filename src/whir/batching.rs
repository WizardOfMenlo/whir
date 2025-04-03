#![cfg(feature = "batching")]

///
/// High level idea of batching multiple polynomials is as follows:
///
/// - Prover commits to multiple Merkle roots of evaluations of each polynomial
///
/// - Verifier samples a batching_randomness field element. For soundness
///   reasons, batching_randomness must be sampled after the Merkle roots have
///   been committed.
///
/// - Prover computes the weighted sum of each individual polynomial based on
///   batching_randomness to compute a single polynomial to proceed during
///   Sumcheck rounds and any other future round.
///
/// - In the first round of the proof, Prover includes the STIR queries from
///   each individual oracle. (The STIR query indexes should be the same for all
///   oracles.)
///
/// - During proof verification, for the first round, the Verifier validates
///   each individual STIR query in the proof against individual Merkle root.
///   Once these Merkle paths are validated, the Verifier re-derives
///   batching_randomness and combines the STIR responses using powers of
///   batching_randomness.
///
/// - After the first round, rest of the protocol proceeds as usual.
///
use ark_crypto_primitives::merkle_tree::{
    Config as MTConfig, LeafParam, MerkleTree, MultiPath, TwoToOneParam,
};
use ark_ff::{FftField, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
#[cfg(any(feature = "parallel", feature = "rayon"))]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{
        ByteDomainSeparator, BytesToUnitDeserialize, BytesToUnitSerialize, FieldDomainSeparator,
        FieldToUnitDeserialize, FieldToUnitSerialize, UnitToField,
    },
    ProofError, ProofResult, UnitToBytes,
};
use spongefish_pow::PoWChallenge;

use crate::{
    fs_utils::{OODDomainSeparator, WhirPoWDomainSeparator},
    poly_utils::coeffs::CoefficientList,
    sumcheck::SumcheckSingleDomainSeparator,
    whir::{
        committer::{CommitmentReader, CommitmentWriter, Witness},
        domainsep::{DigestDomainSeparator, WhirDomainSeparator},
        parameters::WhirConfig,
        prover::Prover,
        statement::{Statement, StatementVerifier},
        utils::{DigestToUnitDeserialize, DigestToUnitSerialize},
        verifier::Verifier,
        RoundZeroProofValidator, WhirCommitmentData, WhirProof, WhirProofRoundData,
    },
};

fn fma_stir_queries<F: Field>(
    scale_factor: F,
    stir_queries: &[Vec<F>],
    folded_queries: &mut [Vec<F>],
) {
    for (folded_vec, stir_vec) in folded_queries.iter_mut().zip(stir_queries.iter()) {
        for (i, value) in stir_vec.iter().enumerate() {
            folded_vec[i] += scale_factor * value
        }
    }
}

#[cfg(debug_assertions)]
fn validate_stir_queries<F, MerkleConfig>(
    batching_randomness: &F,
    expected_stir_value: &[Vec<F>],
    stir_list: &[(MultiPath<MerkleConfig>, Vec<Vec<F>>)],
) -> bool
where
    F: Field,
    MerkleConfig: MTConfig<Leaf = [F]>,
{
    if stir_list.is_empty() {
        return true;
    }

    let (basic_indexes, mut computed_stir) = stir_list[0].clone();
    let mut multiplier = *batching_randomness;

    for (multi_path, stir_queries) in &stir_list[1..] {
        if !multi_path
            .leaf_indexes
            .iter()
            .zip(basic_indexes.leaf_indexes.iter())
            .all(|(lhs, rhs)| lhs == rhs)
        {
            return false;
        }

        fma_stir_queries(multiplier, stir_queries.as_slice(), &mut computed_stir);
        multiplier *= batching_randomness;
    }

    let result = expected_stir_value == computed_stir;
    result
}

#[cfg(not(debug_assertions))]
fn validate_stir_queries<F, MerkleConfig>(
    _batching_randomness: &F,
    _expected_stir_value: &[Vec<F>],
    _stir_list: &[(MultiPath<MerkleConfig>, Vec<Vec<F>>)],
) -> bool
where
    F: Field,
    MerkleConfig: MTConfig<Leaf = [F]>,
{
    true
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchedWhirProof<MerkleConfig, F>
where
    MerkleConfig: MTConfig<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    ///
    /// List of Merkle paths and the PCP slot value for each individual Merkle
    /// tree in the batch. The order of MultiPath must match the order of roots
    /// in the commitment.
    ///
    pub first_round_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,

    /// List of Merkle Paths after first batched round
    pub merkle_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,

    /// Statement value at a random point.
    pub statement_values_at_random_point: Vec<F>,
}

///
/// This is the list of [Witness] structs as if each polynomial will be folded
/// independently. The witnesses are individually committed to the FS-transcript.
///
pub struct BatchedWitness<F, MerkleConfig: MTConfig> {
    pub(crate) witness_list: Vec<Witness<F, MerkleConfig>>,
    pub(crate) batching_randomness: F,
}

impl<F, MerkleConfig> BatchedWitness<F, MerkleConfig>
where
    MerkleConfig: MTConfig,
    F: Field,
{
    ///
    /// Combines all polynomials and leaf nodes weighted by powers of
    /// batching_randomness and returns the pair (batched_polynomial,
    /// batched_oracle). This function is expensive O(2^num_variables).
    ///
    pub fn batch_witnesses(&self) -> (CoefficientList<F>, Vec<F>) {
        assert!(
            self.witness_list.len() > 1,
            "Batched witness list should have at least 2 elements"
        );

        assert!(self.witness_list.iter().all(|wit| wit
            .polynomial
            .coeffs()
            .len()
            .is_power_of_two()
            && wit.merkle_leaves.len().is_power_of_two()
            && wit.polynomial.num_variables() == self.witness_list[0].polynomial.num_variables()
            && wit.merkle_leaves.len() == self.witness_list[0].merkle_leaves.len()),);

        let poly_dim = self.witness_list[0].polynomial.coeffs().len();
        let oracle_len = self.witness_list[0].merkle_leaves.len();
        assert!(oracle_len % poly_dim == 0);
        let rate_factor = oracle_len / poly_dim;

        let mut batched_poly: Vec<F> = self.witness_list[0].polynomial.coeffs().to_vec();
        let mut batched_oracle: Vec<F> = self.witness_list[0].merkle_leaves.clone();

        let mut multiplier = self.batching_randomness;

        // This could potentially be parallelized.
        for wit in &self.witness_list[1..] {
            for (i, coefficient) in wit.polynomial.coeffs().iter().enumerate() {
                batched_poly[i] += multiplier * coefficient;

                for j in 0..rate_factor {
                    let ndx = i + j * poly_dim;
                    batched_oracle[ndx] += multiplier * wit.merkle_leaves[ndx];
                }
            }

            multiplier *= self.batching_randomness;
        }
        (CoefficientList::new(batched_poly), batched_oracle)
    }

    pub fn roots(&self) -> Vec<MerkleConfig::InnerDigest> {
        self.witness_list
            .iter()
            .map(|w| w.merkle_tree.root())
            .collect()
    }
}

pub trait BatchedDomainSeparator<F, MerkleConfig>: WhirDomainSeparator<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: MTConfig,
{
    #[must_use]
    fn batch_commit_statement<PowStrategy>(
        self,
        batch_size: u32,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;
}

impl<F, MerkleConfig, DomainSeparator> BatchedDomainSeparator<F, MerkleConfig> for DomainSeparator
where
    F: FftField,
    MerkleConfig: MTConfig,
    DomainSeparator: ByteDomainSeparator
        + FieldDomainSeparator<F>
        + SumcheckSingleDomainSeparator<F>
        + WhirPoWDomainSeparator
        + OODDomainSeparator<F>
        + DigestDomainSeparator<MerkleConfig>,
{
    ///
    ///  FS Batch Commitment:
    ///  P -> V
    ///     32-bit (4-bytes): Number of entries in the batch
    ///     For each batch entry:
    ///         Merkle Root of prover-id.
    ///
    ///  V -> P
    ///     Sample single scalar element
    ///
    fn batch_commit_statement<PowStrategy>(
        self,
        batch_size: u32,
        _params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        // Add the number of elements that have been batched
        let mut this = self.add_bytes(4, "batch_size");

        for i in 0..batch_size {
            let label = format!("commitment-root-{}", &i);

            // Add the root of each Merkle tree
            this = this.add_digest(&label);

            // TODO: Add the OOD samples if present
            // if params.committment_ood_samples > 0 {
            //     assert!(params.initial_statement);
            //     this = this.add_ood(params.committment_ood_samples);
            // }
        }

        // Sample the batching combination randomness
        this.challenge_scalars(1, "batching_randomness")
    }
}

impl<F, MerkleConfig, PowStrategy> CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: MTConfig<Leaf = [F]>,
{
    pub fn batch_commit<ProverFSState>(
        &self,
        prover_state: &mut ProverFSState,
        polynomial_list: &[CoefficientList<F::BasePrimeField>],
    ) -> ProofResult<BatchedWitness<F, MerkleConfig>>
    where
        ProverFSState: FieldToUnitSerialize<F>
            + BytesToUnitSerialize
            + DigestToUnitSerialize<MerkleConfig>
            + UnitToField<F>,
    {
        assert!(
            polynomial_list.len() > 1,
            "Batched evaluation requires more than one polynomial"
        );

        assert!(
            polynomial_list.iter().all(|poly| {
                poly.num_variables() == polynomial_list[0].num_variables()
                    && poly.num_coeffs() == polynomial_list[0].num_coeffs()
            }),
            "Each polynomial must have same number of variables and same number of coefficients"
        );

        let batch_size = u32::try_from(polynomial_list.len()).unwrap().to_le_bytes();
        prover_state.add_bytes(&batch_size)?;

        let mut witness_list = Vec::new();

        for poly in polynomial_list {
            let individual_commitment = CommitmentWriter::commit(self, prover_state, poly.clone())?;
            witness_list.push(individual_commitment);
        }

        let [batching_randomness] = prover_state.challenge_scalars()?;

        Ok(BatchedWitness {
            witness_list,
            batching_randomness,
        })
    }
}

impl<F, MerkleConfig, PowStrategy> Prover<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: MTConfig<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    fn to_multiroot_proof(
        &self,
        batched_proof: WhirProof<MerkleConfig, F>,
        commitment: BatchedWitness<F, MerkleConfig>,
    ) -> ProofResult<BatchedWhirProof<MerkleConfig, F>> {
        assert!(
            batched_proof.merkle_paths.len() > 0,
            "There must be at least one merkle path"
        );

        let fold_sz = 1 << self.0.folding_factor.at_round(0);
        let mut first_round_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)> =
            Vec::with_capacity(commitment.witness_list.len());

        let ((batched_mt, batched_stir), rest) =
            batched_proof.merkle_paths.as_slice().split_first().unwrap();
        let stir_indexes = batched_mt.leaf_indexes.clone();

        for merkle_tree in commitment.witness_list {
            let paths = merkle_tree
                .merkle_tree
                .generate_multi_proof(stir_indexes.clone())
                .map_err(|_| ProofError::InvalidProof)?;

            let answers: Vec<_> = stir_indexes
                .iter()
                .map(|i| merkle_tree.merkle_leaves[i * fold_sz..(i + 1) * fold_sz].to_vec())
                .collect();

            first_round_paths.push((paths, answers));
        }

        assert!(
            validate_stir_queries(
                &commitment.batching_randomness,
                &batched_stir,
                &first_round_paths,
            ),
            "The computed value of STIR queries must match the value in the leaf nodes"
        );

        Ok(BatchedWhirProof {
            first_round_paths,
            merkle_paths: rest.into(),
            statement_values_at_random_point: batched_proof.statement_values_at_random_point,
        })
    }

    pub fn batch_prove<ProverState>(
        &self,
        prover_state: &mut ProverState,
        statement: Statement<F>,
        witnesses: BatchedWitness<F, MerkleConfig>,
    ) -> ProofResult<BatchedWhirProof<MerkleConfig, F>>
    where
        ProverState: UnitToField<F>
            + FieldToUnitSerialize<F>
            + UnitToBytes
            + spongefish_pow::PoWChallenge
            + DigestToUnitSerialize<MerkleConfig>,
    {
        assert!(self.validate_parameters() && self.validate_statement(&statement));

        assert!(
            witnesses.witness_list.len() > 1
                && witnesses
                    .witness_list
                    .iter()
                    .all(|w| self.validate_witness(w))
        );

        let fold_size = 1 << self.0.folding_factor.at_round(0);
        let (batched_poly, batched_oracle) = witnesses.batch_witnesses();

        // Chunk evaluations into leaves for Merkle tree construction.
        #[cfg(not(feature = "parallel"))]
        let leafs_iter = batched_oracle.chunks_exact(fold_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = batched_oracle.par_chunks_exact(fold_size);

        // Construct the a fake Merkle tree. This is strictly not necessary
        let batched_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        let batched_witness = Witness {
            polynomial: batched_poly,
            merkle_leaves: batched_oracle,
            merkle_tree: batched_tree,
            ood_points: vec![],
            ood_answers: vec![],
        };

        let proof = self.prove(prover_state, statement, batched_witness)?;
        self.to_multiroot_proof(proof, witnesses)
    }
}

#[derive(Clone)]
pub struct ParsedBatchCommitment<F, D> {
    root: Vec<D>,
    batching_randomness: F,
    ood_points: Vec<F>,
    ood_answers: Vec<F>,
}

impl<F, M: MTConfig> WhirCommitmentData<F, M> for ParsedBatchCommitment<F, M::InnerDigest>
where
    F: Clone,
{
    ///
    /// The committed root returned is the root of the first tree. This is just
    /// a dummy value.
    ///
    fn committed_root(&self) -> &<M as MTConfig>::InnerDigest {
        &self.root[0]
    }

    fn ood_data(&self) -> (&[F], &[F]) {
        (&self.ood_points, &self.ood_answers)
    }

    fn batching_randomness(&self) -> Option<F> {
        Some(self.batching_randomness.clone())
    }
}

impl<F, MerkleConfig> RoundZeroProofValidator<ParsedBatchCommitment<F, MerkleConfig::InnerDigest>>
    for BatchedWhirProof<MerkleConfig, F>
where
    MerkleConfig: MTConfig<Leaf = [F]>,
    F: Sized + Clone + ark_serialize::CanonicalSerialize + ark_serialize::CanonicalDeserialize,
{
    type MerkleConfig = MerkleConfig;
    fn validate_first_round(
        &self,
        commitment: &ParsedBatchCommitment<F, MerkleConfig::InnerDigest>,
        stir_challenges_indexes: &[usize],
        leaf_hash: &LeafParam<MerkleConfig>,
        inner_node_hash: &TwoToOneParam<MerkleConfig>,
    ) -> bool {
        commitment.root.len() == self.first_round_paths.len()
            && self
                .first_round_paths
                .iter()
                .zip(commitment.root.iter())
                .all(|((path, answers), root_hash)| {
                    path.verify(
                        leaf_hash,
                        inner_node_hash,
                        root_hash,
                        answers.iter().map(|a| a.as_ref()),
                    )
                    .unwrap()
                        && path.leaf_indexes == *stir_challenges_indexes
                })
    }
}

impl<F, MerkleConfig> WhirProofRoundData<F, MerkleConfig> for BatchedWhirProof<MerkleConfig, F>
where
    MerkleConfig: MTConfig<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize + Field,
{
    //
    // The Merkle paths for round-0 is the path of first oracle. Use
    // RoundZeroProofValidator trait to validate the 0-th round. The leaf
    // node data is combined using the powers of batching randomness
    //
    fn round_data(
        &self,
        whir_round: usize,
        batching_randomness: Option<F>,
    ) -> (MultiPath<MerkleConfig>, Vec<Vec<F>>) {
        assert!(whir_round <= self.merkle_paths.len());
        if whir_round == 0 {
            let batching_randomness = batching_randomness.unwrap();
            let mut multiplier = batching_randomness;
            let (merkle_paths, mut result) = self.first_round_paths[0].clone();

            for (_, leaf) in &self.first_round_paths[1..] {
                fma_stir_queries(multiplier, leaf.as_slice(), result.as_mut_slice());
                multiplier *= batching_randomness;
            }

            (merkle_paths, result)
        } else {
            self.merkle_paths[whir_round - 1].clone()
        }
    }

    fn statement_values(&self) -> &[F] {
        &self.statement_values_at_random_point
    }
}

impl<'a, F, MerkleConfig, PowStrategy> Verifier<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: MTConfig<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub fn verify_batched<VerifierState>(
        &self,
        verifier_state: &mut VerifierState,
        parsed_commitment: &ParsedBatchCommitment<F, MerkleConfig::InnerDigest>,
        statement: &StatementVerifier<F>,
        whir_proof: &BatchedWhirProof<MerkleConfig, F>,
    ) -> ProofResult<()>
    where
        VerifierState: UnitToBytes
            + UnitToField<F>
            + FieldToUnitDeserialize<F>
            + PoWChallenge
            + DigestToUnitDeserialize<MerkleConfig>
            + BytesToUnitDeserialize,
    {
        // Parse proof and validate it STIR queries against the committed Merkle roots.
        let parsed_proof = self.parse_proof(
            verifier_state,
            parsed_commitment,
            whir_proof,
            statement.constraints.len(),
        )?;

        // Verify generically
        self.verify_parsed(statement, parsed_commitment, &parsed_proof)
    }
}

impl<'a, F, MerkleConfig, PowStrategy> CommitmentReader<'a, F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: MTConfig<Leaf = [F]>,
    PowStrategy: spongefish_pow::PowStrategy,
{
    pub fn parse_batched_commitment<VerifierState>(
        &self,
        expected_batch_size: usize,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<ParsedBatchCommitment<F, MerkleConfig::InnerDigest>>
    where
        VerifierState: UnitToBytes
            + FieldToUnitDeserialize<F>
            + UnitToField<F>
            + DigestToUnitDeserialize<MerkleConfig>
            + BytesToUnitDeserialize,
    {
        let mut batch_size_le_bytes: [u8; 4] = [0, 0, 0, 0];
        verifier_state.fill_next_bytes(&mut batch_size_le_bytes)?;
        let batch_size = u32::from_le_bytes(batch_size_le_bytes) as usize;

        if batch_size != expected_batch_size {
            return Err(ProofError::InvalidProof);
        }

        let mut roots = Vec::<MerkleConfig::InnerDigest>::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Order is important here
            let root = verifier_state.read_digest()?;
            roots.push(root);
        }

        // Re-derive batching randomness

        let [batching_randomness] = verifier_state.challenge_scalars()?;

        Ok(ParsedBatchCommitment {
            root: roots,
            batching_randomness,
            ood_points: vec![],
            ood_answers: vec![],
        })
    }
}

#[cfg(test)]
mod batching_tests {
    use ark_ff::Field;
    use ark_std::UniformRand;
    use spongefish::DomainSeparator;
    use spongefish_pow::blake3::Blake3PoW;

    use super::*;
    use crate::{
        crypto::{
            fields::Field64,
            merkle_tree::{
                blake3::{Blake3Compress, Blake3LeafHash, Blake3MerkleTreeParams},
                parameters::default_config,
            },
        },
        parameters::{
            FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
        },
        poly_utils::{
            coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint,
        },
        whir::{
            committer::CommitmentWriter,
            domainsep::WhirDomainSeparator,
            parameters::WhirConfig,
            prover::Prover,
            statement::{Statement, StatementVerifier, Weights},
            verifier::Verifier,
        },
    };

    /// Merkle tree configuration type for commitment layers.
    type MerkleConfig = Blake3MerkleTreeParams<F>;
    /// PoW strategy used for grinding challenges in Fiat-Shamir transcript.
    type PowStrategy = Blake3PoW;
    /// Field type used in the tests.
    type F = Field64;

    /// Run a complete WHIR STARK proof lifecycle: commit, prove, and verify.
    ///
    /// This function:
    /// - builds a multilinear polynomial with a specified number of variables,
    /// - constructs a STARK statement with constraints based on evaluations and linear relations,
    /// - commits to the polynomial using a Merkle-based commitment scheme,
    /// - generates a proof using the WHIR prover,
    /// - verifies the proof using the WHIR verifier.
    fn make_batched_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) {
        println!("Test parameters: ");
        println!("  num_variables  : {}", num_variables);
        println!("  folding_factor : {:?}", &folding_factor);
        println!("  num_points     : {:?}", num_points);
        println!("  soundness_type : {:?}", soundness_type);
        println!("  pow_bits       : {}", pow_bits);
        println!("  fold_type      : {}", fold_type);
        println!("");

        // Number of coefficients in the multilinear polynomial (2^num_variables)
        let num_coeffs = 1 << num_variables;

        // Randomness source
        let mut rng = ark_std::test_rng();
        // Generate Merkle parameters: hash function and compression function
        let (leaf_hash_params, two_to_one_params) =
            default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);

        // Configure multivariate polynomial parameters
        let mv_params = MultivariateParameters::new(num_variables);

        // Configure the WHIR protocol parameters
        let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            leaf_hash_params,
            two_to_one_params,
            soundness_type,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: 1,
            fold_optimisation: fold_type,
            enable_batching: true,
        };

        // Build global configuration from multivariate + protocol parameters
        let params = WhirConfig::new(mv_params, whir_params);

        // Define the multilinear polynomial: constant 1 across all inputs
        let poly1 = CoefficientList::new(vec![F::ONE; num_coeffs]);

        // Define the multilinear polynomial: random
        let poly2 = CoefficientList::new(vec![F::rand(&mut rng); num_coeffs]);

        // Construct a coefficient vector for linear sumcheck constraint
        let weight_poly = CoefficientList::new((0..1 << num_variables).map(F::from).collect());

        // Generate `num_points` random points in the multilinear domain
        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        // Define the Fiat-Shamir IOPattern for committing and proving
        let io = DomainSeparator::new("🌪️")
            .batch_commit_statement(2, &params)
            .add_whir_proof(&params);

        // println!("IO Domain Separator: {:?}", str::from_utf8(io.as_bytes()));
        // Initialize the Merlin transcript from the IOPattern
        let mut prover_state = io.to_prover_state();

        // Create a commitment to the polynomial and generate auxiliary witness data
        let committer = CommitmentWriter::new(params.clone());
        let batched_witness = committer
            .batch_commit(&mut prover_state, &[poly1, poly2])
            .unwrap();

        // Get the batched polynomial
        let (batched_poly, _) = batched_witness.batch_witnesses();

        // Initialize a statement with no constraints yet
        let mut statement = Statement::new(num_variables);

        // For each random point, evaluate the polynomial and create a constraint
        for point in &points {
            let eval = batched_poly.evaluate(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        // Define weights for linear combination
        let linear_claim_weight = Weights::linear(weight_poly.into());

        // Convert polynomial to extension field representation
        let poly = EvaluationsList::from(batched_poly.clone().to_extension());

        // Compute the weighted sum of the polynomial (for sumcheck)
        let sum = linear_claim_weight.weighted_sum(&poly);

        // Add linear constraint to the statement
        statement.add_constraint(linear_claim_weight, sum);

        // Instantiate the prover with the given parameters
        let prover = Prover(params.clone());

        // Extract verifier-side version of the statement (only public data)
        let statement_verifier = StatementVerifier::from_statement(&statement);

        // Generate a STARK proof for the given statement and witness
        let proof = prover
            .batch_prove(&mut prover_state, statement, batched_witness)
            .unwrap();

        // Create a verifier with matching parameters
        let verifier = Verifier::new(&params);

        // Reconstruct verifier's view of the transcript using the IOPattern and prover's data
        let mut verifier_state = io.to_verifier_state(prover_state.narg_string());

        // Create a commitment reader
        let commitment_reader = CommitmentReader::new(&params);

        let parsed_commitment = commitment_reader
            .parse_batched_commitment(2, &mut verifier_state)
            .unwrap();

        // Verify that the generated proof satisfies the statement
        assert!(verifier
            .verify_batched(
                &mut verifier_state,
                &parsed_commitment,
                &statement_verifier,
                &proof
            )
            .is_ok());
    }

    #[test]
    fn test_whir() {
        let folding_factors = [4, 3, 2, 1];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let fold_types = [FoldType::Naive, FoldType::ProverHelps];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_batched_whir_things(
                                    num_variable,
                                    FoldingFactor::Constant(folding_factor),
                                    num_points,
                                    soundness_type,
                                    pow_bits,
                                    fold_type,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
