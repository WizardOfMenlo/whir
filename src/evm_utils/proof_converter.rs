use crate::{
    crypto::{
        fields::{self},
        merkle_tree::keccak::KeccakDigest,
    },
    evm_utils::hasher::MerkleTreeParamsSorted,
};
use ark_crypto_primitives::merkle_tree::{MerkleTree, MultiPath};
use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq)]
pub struct OpenZeppelinMultiProof {
    pub(crate) leaves: Vec<KeccakDigest>,
    pub(crate) proof: Vec<KeccakDigest>,
    pub(crate) proof_flags: Vec<bool>,
    pub(crate) root: KeccakDigest,
}

pub fn convert_proof(
    proof: &MultiPath<
        MerkleTreeParamsSorted<ark_ff::Fp<ark_ff::MontBackend<fields::BN254Config, 4>, 4>>,
    >,
    leaves: Vec<KeccakDigest>,
    root: KeccakDigest,
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
        leaves,
        proof: converted_proof,
        proof_flags: converted_proof_flags,
        root,
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
        MerkleTreeParamsSorted<ark_ff::Fp<ark_ff::MontBackend<fields::BN254Config, 4>, 4>>,
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rayon::slice::ParallelSlice;
    use std::collections::HashSet;
    use std::{error::Error, fs::File, io::Write};

    use crate::{
        crypto::fields::{self, Field256},
        evm_utils::{
            hasher::{sorted_hasher_config, MerkleTreeParamsSorted},
            proof_converter::{convert_proof, get_leaf_hashes, OpenZeppelinMultiProof},
        },
    };

    pub type F = Field256;

    // For simplicity, the leaf preimages are the integers from 1 to 2^tree_height
    #[test]
    fn test_convert_proof_height_1() {
        let tree_height = 1;
        let leaf_preimages: Vec<F> = (1..((1 << tree_height) + 1)).map(|x| F::from(x)).collect();
        let leaves_iter = leaf_preimages.par_chunks_exact(1);
        let mut indices_to_prove = vec![0, 1];
        do_test(
            leaves_iter.clone(),
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_1.json",
        );
    }

    #[test]
    fn test_convert_proof_height_10() {
        let tree_height = 10;
        let leaf_preimages: Vec<F> = (1..((1 << tree_height) + 1)).map(|x| F::from(x)).collect();
        let leaves_iter = leaf_preimages.par_chunks_exact(1);
        let mut indices_to_prove = vec![214];
        do_test(
            leaves_iter.clone(),
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_1.json",
        );
        let mut indices_to_prove = vec![277, 587, 232, 630, 827, 132, 908, 701, 534, 932];
        do_test(
            leaves_iter.clone(),
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_10.json",
        );
        let mut indices_to_prove = vec![
            768, 433, 536, 784, 430, 575, 877, 166, 974, 500, 829, 458, 824, 48, 415, 410, 909,
            210, 485, 515, 384, 203, 1006, 475, 946, 712, 73, 627, 492, 283, 903, 819, 596, 871,
            678, 808, 511, 379, 269, 538, 582, 393, 545, 331, 588, 762, 640, 177, 171, 854, 984,
            884, 666, 619, 891, 687, 912, 746, 83, 684, 910, 212, 683, 437, 883, 835, 539, 240, 75,
            706, 462, 459, 438, 556, 498, 488, 633, 175, 260, 370, 272, 337, 242, 199, 409, 274,
            26, 918, 88, 192, 532, 562, 205, 725, 676,
        ];

        do_test(
            leaves_iter,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_10_100.json",
        );
    }

    #[test]
    fn test_convert_proof_height_20() {
        let tree_height = 20;
        let leaf_preimages: Vec<F> = (1..((1 << tree_height) + 1)).map(|x| F::from(x)).collect();
        let leaves_iter = leaf_preimages.par_chunks_exact(1);
        let mut indices_to_prove = vec![16294];
        do_test(
            leaves_iter.clone(),
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_1.json",
        );
        let mut indices_to_prove = vec![
            872943, 281386, 781188, 158958, 816572, 929118, 24521, 911937, 473111, 85764,
        ];
        do_test(
            leaves_iter.clone(),
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_10.json",
        );
        let mut indices_to_prove = vec![
            748994, 1033416, 271006, 636135, 862066, 527436, 300841, 653560, 408970, 988140,
            708214, 595079, 295725, 587712, 366879, 573675, 526132, 789254, 675307, 254198, 596511,
            527140, 383068, 119305, 99704, 1042764, 617841, 321203, 885103, 79982, 588632, 1040040,
            568447, 184033, 382066, 569274, 1023696, 772943, 252569, 796627, 715993, 631805,
            360758, 879133, 674315, 1000598, 372658, 58816, 132103, 476617, 1046157, 1002958,
            677597, 551978, 546835, 222836, 848765, 59917, 1033276, 80003, 519660, 358965, 640174,
            221549, 634166, 679292, 895295, 574928, 820779, 915242, 183438, 343921, 1043665,
            907102, 853569, 632912, 945061, 571527, 892450, 685559, 638504, 933075, 657470, 154134,
            372927, 143757, 869576, 477214, 716381, 1016022, 618421, 776932, 338531, 238972,
            176408, 641002, 550129, 864620, 116431, 734462,
        ];
        do_test(
            leaves_iter,
            &mut indices_to_prove,
            "src/evm_utils/data/merkle_proof_output_20_100.json",
        );
    }

    fn do_test(
        leaves_iter: rayon::slice::ChunksExact<'_, F>,
        indices_to_prove: &mut [usize],
        file_path: &str,
    ) {
        let merkle_tree =
            MerkleTree::<MerkleTreeParamsSorted<F>>::new(&(), &(), leaves_iter).unwrap();
        indices_to_prove.sort();
        let proof = merkle_tree
            .generate_multi_proof(indices_to_prove.to_owned())
            .unwrap();
        let leaves = get_leaf_hashes(&merkle_tree, indices_to_prove);
        let converted_proof = convert_proof(&proof, leaves, merkle_tree.root());
        let file = File::open(file_path).unwrap();
        let expected_proof: OpenZeppelinMultiProof = serde_json::from_reader(file).unwrap();
        assert_eq!(converted_proof, expected_proof);
    }

    // TODO: this is illustrative code, but it's too specific to dedicate a whole example to it; where
    // to move this?
    fn generate_test_proof() -> Result<(), Box<dyn Error>> {
        use fields::Field256 as F;
        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = sorted_hasher_config::<F>(&mut rng);
        let tree_height = 10;

        // For simplicity, the leaf preimages are the integers from 1 to 2^tree_height
        let leaf_preimages: Vec<F> = (1..((1 << tree_height) + 1)).map(|x| F::from(x)).collect();
        let leaves_iter = leaf_preimages.par_chunks_exact(1);
        let merkle_tree = MerkleTree::<MerkleTreeParamsSorted<F>>::new(
            &leaf_hash_params,
            &two_to_one_params,
            leaves_iter,
        )
        .unwrap();
        let mut tree_rng = StdRng::from_entropy();

        // Let's select ~10 random unique indices to prove
        let indices_to_prove: Vec<usize> = (0..100)
            .map(|_| tree_rng.gen_range(0..(1 << tree_height)))
            .collect();

        // Make sure that all indices are unique
        let indices_to_prove_set = HashSet::<usize>::from_iter(indices_to_prove.iter().cloned());
        let indices_to_prove: Vec<usize> = indices_to_prove_set.iter().cloned().collect();
        println!("Indices to prove: {:?}", indices_to_prove);
        let mut indexes = indices_to_prove.clone();
        indexes.sort();

        let proof = merkle_tree.generate_multi_proof(indexes.clone()).unwrap();
        let leaves = get_leaf_hashes(&merkle_tree, &indexes);
        let converted_proof = convert_proof(&proof, leaves, merkle_tree.root());

        let json_string = serde_json::to_string_pretty(&converted_proof)?;
        let mut file = File::create("proof_output.json")?;
        file.write_all(json_string.as_bytes())?;
        println!("Proof successfully written to `proof_output.json`.");

        Ok(())
    }
}
