use ark_crypto_primitives::merkle_tree::{Config, LeafParam, MerkleTree, MultiPath, TwoToOneParam};
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::{DuplexSpongeInterface, VerificationError, VerificationResult};

use crate::{
    crypto::merkle_tree::proof::FullMultiPath,
    parameters::MerkleProofStrategy,
    transcript::{ProverState, VerifierState},
};

/// Prover-side state for emitting Merkle proofs using a fixed strategy.
pub struct ProverMerkleState {
    strategy: MerkleProofStrategy,
}

impl ProverMerkleState {
    pub const fn new(strategy: MerkleProofStrategy) -> Self {
        Self { strategy }
    }

    pub fn write_proof_hint<H, R, MerkleConfig>(
        &self,
        tree: &MerkleTree<MerkleConfig>,
        indices: &[usize],
        prover_state: &mut ProverState<H, R>,
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        MerkleConfig: Config,
    {
        match self.strategy {
            MerkleProofStrategy::Compressed => {
                let merkle_proof = tree
                    .generate_multi_proof(indices.to_vec())
                    .expect("indices sampled from transcript must be valid for the tree");
                prover_state.prover_hint_ark(&merkle_proof);
            }
            MerkleProofStrategy::Uncompressed => {
                let proofs = indices
                    .iter()
                    .map(|&index| {
                        tree.generate_proof(index)
                            .expect("indices sampled from transcript must be valid for the tree")
                    })
                    .collect::<Vec<_>>();
                let merkle_proof: FullMultiPath<MerkleConfig> = proofs.into();

                prover_state.prover_hint_ark(&merkle_proof);
            }
        }
    }
}

/// Verifier-side state for consuming Merkle proofs using a fixed strategy and hash parameters.
pub struct VerifierMerkleState<'a, MerkleConfig>
where
    MerkleConfig: Config,
{
    strategy: MerkleProofStrategy,
    leaf_hash_params: &'a LeafParam<MerkleConfig>,
    two_to_one_params: &'a TwoToOneParam<MerkleConfig>,
}

impl<'a, MerkleConfig> VerifierMerkleState<'a, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub const fn new(
        strategy: MerkleProofStrategy,
        leaf_hash_params: &'a LeafParam<MerkleConfig>,
        two_to_one_params: &'a TwoToOneParam<MerkleConfig>,
    ) -> Self {
        Self {
            strategy,
            leaf_hash_params,
            two_to_one_params,
        }
    }

    pub fn read_and_verify_proof<H, L>(
        &self,
        verifier_state: &mut VerifierState<H>,
        indices: &[usize],
        root: &MerkleConfig::InnerDigest,
        leaves: impl IntoIterator<Item = L>,
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        L: Clone + std::borrow::Borrow<MerkleConfig::Leaf>,
    {
        let leaves: Vec<L> = leaves.into_iter().collect();

        match self.strategy {
            MerkleProofStrategy::Compressed => {
                let merkle_proof: MultiPath<MerkleConfig> = verifier_state.prover_hint_ark()?;
                if merkle_proof.leaf_indexes != indices {
                    return Err(VerificationError);
                }

                let correct = merkle_proof
                    .verify(
                        self.leaf_hash_params,
                        self.two_to_one_params,
                        root,
                        leaves.iter().cloned(),
                    )
                    .map_err(|_| VerificationError)?;
                if !correct {
                    return Err(VerificationError);
                }
            }
            MerkleProofStrategy::Uncompressed => {
                let merkle_proof: FullMultiPath<MerkleConfig> = verifier_state.prover_hint_ark()?;
                if merkle_proof.indices() != indices {
                    return Err(VerificationError);
                }

                let correct = merkle_proof
                    .verify(
                        self.leaf_hash_params,
                        self.two_to_one_params,
                        root,
                        leaves.iter().cloned(),
                    )
                    .map_err(|_| VerificationError)?;
                if !correct {
                    return Err(VerificationError);
                }
            }
        }

        Ok(())
    }
}
