use crate::evm_utils::proof_serde::EvmFieldElementSerDe;
use crate::{
    crypto::{
        fields::{self},
        merkle_tree::keccak::KeccakDigest,
    },
    evm_utils::hasher::MerkleTreeEvmParams,
    fs_utils::EVMFs,
    whir::{Statement, WhirProof},
};
use ark_crypto_primitives::merkle_tree::{MerkleTree, MultiPath};
use ark_ff::PrimeField;
use serde::{ser::SerializeStruct, Serialize};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

#[derive(Debug, PartialEq)]
pub struct OpenZeppelinMultiProof {
    pub(crate) proof: Vec<KeccakDigest>,
    pub(crate) proof_flags: Vec<bool>,
}

pub struct FullEvmProof<F: PrimeField> {
    pub whir_proof: WhirEvmProof<F>,
    pub statement: Statement<F>,
    pub arthur: EVMFs<F>,
}

impl<F: PrimeField> Serialize for FullEvmProof<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("FullEvmProof", 3)?;
        let mut proofs_vec = vec![];
        let mut answers_vec = vec![];
        for (proof, answers) in self.whir_proof.0.iter() {
            proofs_vec.push(proof);
            answers_vec.push(
                answers
                    .iter()
                    .map(|inner_vec| inner_vec.iter().map(F::serialize).collect::<Vec<String>>())
                    .collect::<Vec<Vec<String>>>(),
            );
        }
        state.serialize_field("merkleProofs", &proofs_vec)?;
        state.serialize_field("answers", &answers_vec)?;
        state.serialize_field("statement", &self.statement)?;
        state.serialize_field("arthur", &self.arthur)?;
        state.end()
    }
}

pub struct WhirEvmProof<F>(pub(crate) Vec<(OpenZeppelinMultiProof, Vec<Vec<F>>)>);

/// Converts WHIR proof to an EVM-friendly format where the merkle proof is an `OpenZeppelinMultiProof`
pub fn convert_whir_proof<PowStrategy, F: PrimeField>(
    proof: WhirProof<MerkleTreeEvmParams<F>, F>,
) -> Result<WhirEvmProof<F>, Box<dyn Error>> {
    let mut converted_proofs = vec![];
    for (proof, answers) in proof.0 {
        converted_proofs.push((convert_merkle_proof(&proof), answers));
    }
    Ok(WhirEvmProof(converted_proofs))
}

/// Converts a proof from the Arkworks to the OpenZeppelin format
pub fn convert_merkle_proof<F: PrimeField>(
    proof: &MultiPath<MerkleTreeEvmParams<F>>,
) -> OpenZeppelinMultiProof {
    let path_len = proof.auth_paths_suffixes[0].len();
    let tree_height = path_len + 1;

    let mut node_by_tree_index = HashMap::<usize, KeccakDigest>::new();
    let mut calculated_node_tree_indices = HashSet::<usize>::new();

    let mut converted_proof = vec![];
    let mut converted_proof_flags: Vec<bool> = vec![];

    let mut prev_path: Vec<_> = proof.auth_paths_suffixes[0].clone();
    for (i, leaf_idx) in proof.leaf_indexes.iter().enumerate() {
        // Determine if sibling is a part of "main queue" or "proof"
        let sibling_idx = leaf_idx ^ 1;
        if proof.leaf_indexes.contains(&sibling_idx) {
            if sibling_idx > *leaf_idx {
                // For a pair, push the flag only once!
                converted_proof_flags.push(true);
            }
        } else {
            converted_proof.push(proof.leaf_siblings_hashes[i]);
            converted_proof_flags.push(false);
        }

        // Decode the full auth path
        let path = prefix_decode_path(
            &prev_path,
            proof.auth_paths_prefix_lenghts[i],
            &proof.auth_paths_suffixes[i],
        );
        prev_path = path.clone();

        // For each path, determine the indices of the auth path nodes above leaf level
        let mut parent_level_idx = leaf_idx >> 1;
        for level in (1..tree_height).rev() {
            // All the parents along the path will be calculated during the proof verification
            calculated_node_tree_indices.insert(to_tree_index(parent_level_idx, level));
            let parent_sibling_level_idx = parent_level_idx ^ 1;
            // We "cache" the auth path nodes to later pick ones that won't be calculated in any of the paths
            node_by_tree_index.insert(
                to_tree_index(parent_sibling_level_idx, level),
                // The path goes from root to leaves, so we need to reverse
                path[level - 1],
            );
            parent_level_idx >>= 1;
        }
    }

    // Second pass
    for level in (1..path_len + 1).rev() {
        // For each level, find nodes that won't be calculated
        let level_size = 1 << level;
        for i in 0..level_size {
            if calculated_node_tree_indices.contains(&to_tree_index(i, level)) {
                let sibling_idx = i ^ 1;
                let sibling_tree_idx = &to_tree_index(sibling_idx, level);
                if calculated_node_tree_indices.contains(sibling_tree_idx) {
                    // Both siblings are calculated. Adding true flag only once:
                    if sibling_idx > i {
                        converted_proof_flags.push(true);
                    }
                } else if node_by_tree_index.contains_key(sibling_tree_idx) {
                    converted_proof.push(node_by_tree_index[sibling_tree_idx]);
                    converted_proof_flags.push(false);
                }
            }
        }
    }

    OpenZeppelinMultiProof {
        proof: converted_proof,
        proof_flags: converted_proof_flags,
    }
}

fn to_tree_index(level_index: usize, level: usize) -> usize {
    (1 << level) + level_index - 1
}

// Adapted from ark-crypto-primitives (it's private there)
fn prefix_decode_path<T>(prev_path: &[T], prefix_len: usize, suffix: &Vec<T>) -> Vec<T>
where
    T: Eq + Clone,
{
    if prefix_len == 0 {
        suffix.to_owned()
    } else {
        [prev_path[0..prefix_len].to_vec(), suffix.to_owned()].concat()
    }
}

pub fn get_leaf_hashes(
    merkle_tree: &MerkleTree<
        MerkleTreeEvmParams<ark_ff::Fp<ark_ff::MontBackend<fields::BN254Config, 4>, 4>>,
    >,
    indices_to_prove: &[usize],
) -> Vec<KeccakDigest> {
    indices_to_prove
        .iter()
        // Unfortunately, the Arkworks Merkle tree lacks a method to get the leaf hash directly
        // so we get the "sibling of a sibling" hash instead
        .map(|i| merkle_tree.get_leaf_sibling_hash(*i ^ 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use ark_crypto_primitives::merkle_tree::MerkleTree;
    use rayon::slice::ParallelSlice;
    use std::fs::File;

    use crate::{
        crypto::fields::Field256,
        evm_utils::{
            hasher::MerkleTreeEvmParams,
            proof_converter::{convert_merkle_proof, OpenZeppelinMultiProof},
        },
    };

    pub type F = Field256;

    // For simplicity, the leaf preimages are the integers from 1 to 2^tree_height
    #[test]
    fn test_convert_proof_height_1() {
        let mut indices_to_prove = vec![0];
        do_test(
            1,
            1 << 0,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_1.json",
        );
    }

    #[test]
    fn test_convert_proof_height_10() {
        let mut indices_to_prove = vec![44];
        do_test(
            10,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_1.json",
        );
        let mut indices_to_prove = vec![9, 10, 0, 6, 33, 38, 48, 34, 13];
        do_test(
            10,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_10.json",
        );
        let mut indices_to_prove = vec![
            54, 20, 21, 48, 28, 39, 4, 22, 60, 44, 24, 14, 7, 19, 50, 29, 34, 51, 55, 12, 18, 6,
            61, 47, 13, 40, 31, 37, 36, 15, 63, 42, 32, 49, 2, 16, 59, 35, 23, 53, 33, 3, 25, 27,
            38, 52, 62, 1, 26, 56, 43, 45,
        ];

        do_test(
            10,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_100.json",
        );
    }

    #[test]
    fn test_convert_proof_height_20() {
        let mut indices_to_prove = vec![14930];
        do_test(
            20,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_1.json",
        );
        let mut indices_to_prove = vec![
            60301, 60024, 5686, 12933, 22858, 11867, 21210, 53443, 19564, 9905,
        ];
        do_test(
            20,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_10.json",
        );
        let mut indices_to_prove = vec![
            30987, 15226, 39864, 37722, 52110, 50515, 26696, 53234, 14306, 13995, 31263, 2120,
            54428, 29679, 48352, 41995, 14991, 22811, 20680, 22681, 36424, 60764, 48756, 3973,
            17334, 59257, 45702, 3335, 55448, 47882, 17075, 7723, 5241, 23252, 16147, 13186, 35226,
            34402, 6575, 21712, 10983, 8235, 22752, 41025, 20973, 40605, 24739, 41570, 57796,
            41881, 6546, 59343, 10908, 5876, 1903, 11585, 6442, 37180, 6105, 60955, 30246, 49744,
            55660, 63781, 26923, 45871, 22301, 40003, 56655, 14004, 4872, 50156, 41002, 63354,
            12016, 58674, 58865, 41611, 52420, 43373, 29911, 42068, 58172, 4851, 54983, 62655,
            32514, 43383, 62165, 47170, 290, 54989, 48340, 35835, 17857, 28637, 43940, 22401,
            11785, 44568,
        ];
        do_test(
            20,
            1 << 4,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_100.json",
        );
    }

    fn do_test(
        tree_height: usize,
        fold_size: usize,
        indices_to_prove: &mut [usize],
        file_path: &str,
    ) {
        let leaf_preimages: Vec<F> = (1..((1 << tree_height) + 1)).map(|x| F::from(x)).collect();

        let leaves_iter = leaf_preimages.par_chunks_exact(fold_size);
        let merkle_tree = MerkleTree::<MerkleTreeEvmParams<F>>::new(&(), &(), leaves_iter).unwrap();
        indices_to_prove.sort();
        let proof = merkle_tree
            .generate_multi_proof(indices_to_prove.to_owned())
            .unwrap();

        let converted_proof = convert_merkle_proof(&proof);
        let file = File::open(file_path).unwrap();
        let expected_proof: OpenZeppelinMultiProof = serde_json::from_reader(file).unwrap();
        assert_eq!(converted_proof, expected_proof);
    }
}
