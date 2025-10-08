use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use spongefish::{ProofError, ProofResult};
use crate::merkle_tree::{Hash, Hasher, MerkleProof};
use crate::{
    merkle_tree::MerkleTreeHasher, whir::utils::{HintDeserialize, HintSerialize}
};

/// Prover-side state for emitting Merkle proofs using a fixed strategy.
pub struct ProverMerkleState {
}

impl ProverMerkleState {
    pub const fn new() -> Self {
        Self { }
    }

    pub fn write_proof_hint<F, ProverState>(
        &self,
        fold_size: usize,
        previous_merkle_answers: &Vec<F>,
        tree_new: &MerkleTreeHasher,
        indices: &[usize],
        prover_state: &mut ProverState,
    ) -> ProofResult<()>
    where
        F: Clone + CanonicalSerialize + ark_ff::Field,
        ProverState: HintSerialize,
    {
        let new_proof =  tree_new.proof(indices, |i, d| {
            if d == tree_new.depth {
                let leaf : Vec<_> = previous_merkle_answers[i * fold_size..(i + 1) * fold_size].to_vec();
                let mut buf = [0u8; 32];
                let m = leaf.iter().cloned().reduce(|a, b| {
                    let mut l_input = [0u8; 32];
                    let mut r_input = [0u8; 32];
                    a.serialize_uncompressed(&mut l_input[..]).expect("leaf hash failed");
                    b.serialize_uncompressed(&mut r_input[..]).expect("leaf hash failed");
                    let mut out = [[0u8; 32]; 1];
                    tree_new.hasher_at_depth(0).hash_pairs(&[l_input, r_input], &mut out);
                    F::from_base_prime_field(<F::BasePrimeField as ark_ff::PrimeField>::from_le_bytes_mod_order(&out[0]))
                }).unwrap();
                m.serialize_uncompressed(&mut buf[..]).expect("leaf hash failed");
                buf
            } else {
                println!("i: , d: {:?} {:?}", i, d);
                tree_new.layers[d][i]
            }
        });

        prover_state.hint::<MerkleProof<F>>(&new_proof)?;
    
        Ok(())
    }
}

/// Verifier-side state for consuming Merkle proofs using a fixed strategy and hash parameters.
pub struct VerifierMerkleState {
}

impl VerifierMerkleState
{
    pub const fn new() -> Self {
        Self {}
    }

    pub fn read_and_verify_proof<VerifierState, F>(
        &self,
        verifier_state: &mut VerifierState,
        indices: &[usize],
        root: &Hash,
        leaves: &[Hash],
        construct_skyscraper: fn() -> Box<dyn Hasher>,
    ) -> ProofResult<()>
    where
        VerifierState: HintDeserialize,
        F: CanonicalSerialize + CanonicalDeserialize,
    {
        let new_proof: MerkleProof<F> = verifier_state.hint()?;

        println!("depth: {:?}", new_proof.depth);
        // println!("new_proof: {:?}", new_proof.1);
        println!("indices from proof: {:?}", new_proof.indices);
        println!("indices: {:?}", indices);
        if new_proof.indices != indices {
            return Err(ProofError::InvalidProof);
        }

        let merklizer = MerkleTreeHasher::from_hasher_fn(new_proof.depth, construct_skyscraper);

        let correct = merklizer.verify_multi(root, &indices, &leaves, &new_proof.proof.iter().map(|h| {let mut buf = [0u8; 32]; h.serialize_uncompressed(&mut buf[..]).expect("leaf hash failed"); buf}).collect::<Vec<_>>());
        println!("correct: {:?}", correct.clone().err());
        println!("correct: {:?}", correct.is_ok());
        if !correct.is_ok() {
            return Err(ProofError::InvalidProof);
        }
        Ok(())
    }
}
