use ark_crypto_primitives::merkle_tree::Config;
use ark_ff::FftField;
use spongefish::codecs::arkworks_algebra::{ByteDomainSeparator, FieldDomainSeparator};

use super::parameters::WhirConfig;
use crate::{
    fs_utils::{OODDomainSeparator, WhirPoWDomainSeparator},
    sumcheck::SumcheckSingleDomainSeparator,
};

pub trait DigestDomainSeparator<MerkleConfig: Config> {
    #[must_use]
    fn add_digest(self, label: &str) -> Self;
}

pub trait WhirDomainSeparator<F: FftField, MerkleConfig: Config> {
    #[must_use]
    fn commit_statement<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self;

    /// Domain separator for regular single-commitment proving
    #[must_use]
    fn add_whir_proof<PowStrategy>(self, params: &WhirConfig<F, MerkleConfig, PowStrategy>)
        -> Self;

    /// Domain separator for batch proving multiple commitments
    #[must_use]
    fn add_whir_batch_proof<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
        num_witnesses: usize,
        num_constraints_total: usize,
    ) -> Self;

    /// Domain separator for pre-folding a single variable
    /// Used in PreFold approach to fold larger polynomial once before batching
    #[must_use]
    fn add_prefold_single_variable(self) -> Self;

    /// Domain separator for PreFold batch proving
    /// Handles the complete PreFold protocol: matrix commit, fold, batch, and prove
    #[must_use]
    fn add_whir_prefold_batch_proof<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
        num_constraints_total: usize,
    ) -> Self;
}

impl<F, MerkleConfig, DomainSeparator> WhirDomainSeparator<F, MerkleConfig> for DomainSeparator
where
    F: FftField,
    MerkleConfig: Config,
    DomainSeparator:
        ByteDomainSeparator + FieldDomainSeparator<F> + DigestDomainSeparator<MerkleConfig>,
{
    //
    // FS Batch Commitment:
    // P -> V
    //  For each batch entry:
    //      Merkle Root of prover-id.
    //      Sample List of OOD queries based on _all_ committed roots
    //      Compute List of OOD responses for each root
    //
    //  If more than on entry is there add
    //  V -> P
    //      Sample Single field element
    //

    fn commit_statement<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        let mut this = self;
        this = this.add_digest("merkle-root");

        if params.committment_ood_samples > 0 {
            assert!(params.initial_statement);
            this = this.add_ood(params.committment_ood_samples, params.batch_size);
        }

        if params.batch_size > 1 {
            this = this.challenge_scalars(1, "batching_randomness");
        }

        this
    }

    fn add_whir_proof<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    ) -> Self {
        add_whir_proof_impl(self, params, None)
    }

    fn add_whir_batch_proof<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
        num_witnesses: usize,
        num_constraints_total: usize,
    ) -> Self {
        // Step 1: Commit the full N×M constraint evaluation matrix to the transcript.
        // This binds the prover to all cross-term evaluations before sampling γ,
        // preventing adaptive attacks where the prover could choose cross-terms
        // after seeing the batching challenge.
        let matrix_size = num_witnesses * num_constraints_total;
        let this = self.add_scalars(matrix_size, "constraint_evaluation_matrix");

        // Step 2: Sample batching randomness γ after committing evaluations
        let this = this.challenge_scalars(1, "batching_randomness");

        // Step 3: Continue with standard WHIR proof protocol
        add_whir_proof_impl(this, params, Some(num_witnesses))
    }

    fn add_prefold_single_variable(self) -> Self {
        // Domain separation for pre-folding phase
        self.challenge_scalars(1, "prefold_folding_randomness")
            .add_digest("prefold_folded_commitment")
    }

    fn add_whir_prefold_batch_proof<PowStrategy>(
        self,
        params: &WhirConfig<F, MerkleConfig, PowStrategy>,
        num_constraints_total: usize,
    ) -> Self {
        // PreFold batch protocol structure (simplified, 2 witnesses):
        // 1. Sample prefold randomness α and commit to folded polynomial g'
        // 2. Prove consistency of g' with original g via STIR queries on g
        // 3. Commit the 2×M constraint evaluation matrix for (f, g') BEFORE sampling γ
        // 4. Sample batching randomness γ and run standard WHIR on h = f + γ·g'

        // Step 1: Prefold phase - sample 1 random value to fold exactly 1 variable
        let mut this = self.challenge_scalars(1, "prefold_folding_randomness");

        if params.starting_folding_pow_bits > 0. {
            this = this.pow(params.starting_folding_pow_bits);
        }

        // Commit folded polynomial g'
        this = this.add_digest("prefold_folded_commitment");

        // OOD sampling for g'
        if params.committment_ood_samples > 0 {
            this = this.add_ood(params.committment_ood_samples, 1);
        }

        // PoW before STIR queries
        if params.round_parameters[0].pow_bits > 0. {
            this = this.pow(params.round_parameters[0].pow_bits);
        }

        // STIR queries on original g (consistency check)
        // PreFold requires g to be committed with folding_factor=1
        let g_domain_size = params.starting_domain.size() * 2;
        let g_folding_factor = 1; // PreFold requires folding_factor=1 for g
        let folded_domain_size = g_domain_size >> g_folding_factor;
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        this = this.challenge_bytes(
            params.round_parameters[0].num_queries * domain_size_bytes,
            "stir_queries_original_g",
        );

        // STIR answers from original g and merkle proof
        this = this
            .hint("stir_answers_original_g")
            .hint("merkle_proof_original_g");

        // Add g' STIR evaluations
        // Note: Variable length due to deduplication, sent as scalars
        this = this.hint("g_folded_stir_evals");

        // Step 3: Commit the 2×M constraint evaluation matrix for (f, g') BEFORE sampling γ
        let matrix_size = 2 * num_constraints_total;
        this = this.add_scalars(matrix_size, "constraint_evaluation_matrix");

        // Step 4: Sample batching randomness γ
        this = this.challenge_scalars(1, "batching_randomness");

        // Step 5: Continue with standard *batch* WHIR proof on h = f + γ·g'
        // (Round 0 includes 2 Merkle openings: one for f and one for g'.)
        add_whir_proof_impl(this, params, Some(2))
    }
}

/// Private helper: shared implementation for both regular and batch proving.
///
/// # Arguments
/// * `ds` - Domain separator state
/// * `params` - WHIR protocol configuration
/// * `num_witnesses` - `None` for regular proving, `Some(n)` for batch proving with n witnesses
fn add_whir_proof_impl<F, MerkleConfig, DomainSeparator, PowStrategy>(
    mut ds: DomainSeparator,
    params: &WhirConfig<F, MerkleConfig, PowStrategy>,
    num_witnesses: Option<usize>,
) -> DomainSeparator
where
    F: FftField,
    MerkleConfig: Config,
    DomainSeparator:
        ByteDomainSeparator + FieldDomainSeparator<F> + DigestDomainSeparator<MerkleConfig>,
{
    // Initial sumcheck (same for both regular and batch)
    if params.initial_statement {
        ds = ds
            .challenge_scalars(1, "initial_combination_randomness")
            .add_sumcheck(
                params.folding_factor.at_round(0),
                params.starting_folding_pow_bits,
            );
    } else {
        ds = ds
            .challenge_scalars(params.folding_factor.at_round(0), "folding_randomness")
            .pow(params.starting_folding_pow_bits);
    }

    let mut domain_size = params.starting_domain.size();

    // Round handling
    for (round, r) in params.round_parameters.iter().enumerate() {
        let folded_domain_size = domain_size >> params.folding_factor.at_round(round);
        let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

        // Digest label differs for batch round 0
        let digest_label = if round == 0 && num_witnesses.is_some() {
            "batched_merkle_digest" // Round 0 commits to the batched polynomial
        } else {
            "merkle_digest"
        };

        ds = ds
            .add_digest(digest_label)
            .add_ood(r.ood_samples, 1)
            .pow(r.pow_bits)
            .challenge_bytes(r.num_queries * domain_size_bytes, "stir_queries");

        // Round 0 Merkle proofs: batch proving requires N proofs (one per original tree),
        // while regular proving requires just 1 proof.
        if round == 0 {
            if let Some(n) = num_witnesses {
                // Batch proving: verify openings in all N original commitment trees
                for i in 0..n {
                    ds = ds
                        .hint(&format!("stir_answers_witness_{i}"))
                        .hint(&format!("merkle_proof_witness_{i}"));
                }
            } else {
                // Regular proving: single commitment tree
                ds = ds.hint("stir_answers").hint("merkle_proof");
            }
        } else {
            // Rounds 1+: all proving modes use the single batched tree
            ds = ds.hint("stir_answers").hint("merkle_proof");
        }

        ds = ds
            .challenge_scalars(1, "combination_randomness")
            .add_sumcheck(
                params.folding_factor.at_round(round + 1),
                r.folding_pow_bits,
            );
        domain_size >>= 1;
    }

    // Final round (same for both regular and batch)
    let folded_domain_size = domain_size
        >> params
            .folding_factor
            .at_round(params.round_parameters.len());
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize).div_ceil(8);

    ds.add_scalars(1 << params.final_sumcheck_rounds, "final_coeffs")
        .pow(params.final_pow_bits)
        .challenge_bytes(domain_size_bytes * params.final_queries, "final_queries")
        .hint("stir_answers")
        .hint("merkle_proof")
        .add_sumcheck(params.final_sumcheck_rounds, params.final_folding_pow_bits)
        .hint("deferred_weight_evaluations")
}
