//! Protocol for committing to a vector of [`struct@Hash`]es.
//!
//! Uses a mixed-arity Merkle tree: each layer has its own arity (2, 3, or 13).
//! For `num_leaves = 2^a * 3^b * 13^c`, the tree uses `c` arity-13 layers
//! (bottom), `b` ternary layers (middle), and `a` binary layers (top).
//!
//! See <https://eprint.iacr.org/2026/089> for analysis when used with truncated permutation
//! node hashes.

use std::{fmt, mem::swap};

use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::{instrument, span, Level};
use zerocopy::IntoBytes;

use crate::{
    engines::EngineId,
    hash::{self, Hash, HashEngine, ENGINES},
    transcript::{
        DuplexSpongeInterface, ProverMessage, ProverState, VerificationError, VerificationResult,
        VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Size in bytes of the hash input for one internal node with the given arity.
pub const fn node_hash_size(arity: usize) -> usize {
    arity * 32
}

/// Maximum supported arity. Used for fixed-size arrays in open/verify loops.
const MAX_ARITY: usize = 13;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
pub struct Config {
    /// Number of leaves in the Merkle tree.
    pub num_leaves: usize,

    /// Layer configurations for the Merkle tree, root to bottom.
    pub layers: Vec<LayerConfig>,
}

#[derive(
    Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Debug, Hash, Default, Serialize, Deserialize,
)]
pub struct LayerConfig {
    /// The engine used to hash siblings.
    pub hash_id: EngineId,

    /// Arity of this layer (number of children per node). Must be 2, 3, or 13.
    pub arity: usize,
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MerkleTree(num_leaves: {})", self.num_leaves)
    }
}

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize,
)]
#[must_use]
pub struct Commitment {
    /// The commitment root hash.
    hash: Hash,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Witness {
    /// The nodes in the Merkle tree, starting with the leaf hash layer.
    nodes: Vec<Hash>,
}

impl Config {
    /// Create a new Merkle tree configuration with the recommended hash function.
    pub fn new(num_leaves: usize) -> Self {
        Self::with_hash(hash::BLAKE3, num_leaves)
    }

    /// Create a mixed-arity Merkle tree configuration.
    ///
    /// The tree pads `num_leaves` to the smallest smooth-{2,3,13} number >= `num_leaves`.
    pub fn with_hash(hash_id: EngineId, num_leaves: usize) -> Self {
        let arities = layers_for_size(num_leaves);
        let layers = arities
            .iter()
            .map(|&arity| LayerConfig { hash_id, arity })
            .collect();
        Self { num_leaves, layers }
    }

    /// Total number of nodes in the tree (all layers including leaves).
    pub fn num_nodes(&self) -> usize {
        if self.layers.is_empty() {
            return 1;
        }
        let mut total = 0;
        let mut layer_size = 1; // root
                                // Layers are root-to-bottom, so multiply by arity going down.
        for layer in &self.layers {
            layer_size *= layer.arity;
            total += layer_size;
        }
        total + 1 // +1 for the root node
    }

    /// Capacity of the leaf layer (product of all arities).
    fn leaf_capacity(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.arity)
            .product::<usize>()
            .max(1)
    }

    #[cfg_attr(feature = "tracing", instrument(skip(prover_state, leaves), fields(self = %self)))]
    pub fn commit<H, R>(&self, prover_state: &mut ProverState<H, R>, leaves: Vec<Hash>) -> Witness
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            leaves.len(),
            self.num_leaves,
            "Expected {} leaf hashes, got {}",
            self.num_leaves,
            leaves.len()
        );

        // Allocate nodes and fill with leaf layer. This implicitly pads the first layer.
        let mut nodes = leaves;
        nodes.resize(self.num_nodes(), Hash::default());
        let (mut previous, mut remaining) = nodes.split_at_mut(self.leaf_capacity());

        // Compute merkle tree nodes bottom-to-top.
        for layer in self.layers.iter().rev() {
            let arity = layer.arity;
            let parent_count = previous.len() / arity;
            let (current, next_remaining) = remaining.split_at_mut(parent_count);
            let engine = ENGINES
                .retrieve(layer.hash_id)
                .expect("Hash Engine not found");
            #[cfg(feature = "tracing")]
            let _span = span!(
                Level::DEBUG,
                "layer",
                engine = engine.name().as_ref(),
                count = current.len(),
                arity = arity
            )
            .entered();

            // TODO: Parallelize over subtrees, not layerwise. This will
            // increase locality.
            let hash_size = node_hash_size(arity);
            parallel_hash(&*engine, hash_size, previous.as_bytes(), current);
            previous = current;
            remaining = next_remaining;
        }

        // Commit to the root hash.
        prover_state.prover_message(&previous[0]);

        Witness { nodes }
    }

    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        let hash = verifier_state.prover_message()?;
        Ok(Commitment { hash })
    }

    /// Opens the commitment at the provided indices.
    ///
    /// Indices can be in any order and may contain duplicates.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(num_indices = indices.len())))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness,
        indices: &[usize],
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(witness.nodes.len(), self.num_nodes());
        assert!(indices.iter().all(|&i| i < self.num_leaves));

        // Abstract execution of verify algorithm writing required hashes.
        let mut indices = indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        let (mut layer, mut remaining) = witness.nodes.split_at(self.leaf_capacity());

        // Walk bottom-to-top through layers.
        for layer_config in self.layers.iter().rev() {
            let arity = layer_config.arity;
            let mut next_indices = Vec::with_capacity(indices.len());
            let mut iter = indices.iter().copied().peekable();
            loop {
                match iter.next() {
                    Some(a) => {
                        let parent = a / arity;
                        let group_start = parent * arity;

                        // Track which siblings are present in the query set.
                        let mut present = [false; MAX_ARITY];
                        present[a - group_start] = true;
                        while let Some(&b) = iter.peek() {
                            if b / arity == parent {
                                present[b - group_start] = true;
                                iter.next();
                            } else {
                                break;
                            }
                        }

                        // Send the missing siblings as hints.
                        for i in 0..arity {
                            if !present[i] {
                                prover_state.prover_hint(&layer[group_start + i]);
                            }
                        }
                        next_indices.push(parent);
                    }
                    None => break,
                }
            }
            indices = next_indices;
            let parent_count = layer.len() / arity;
            let (next_layer, next_remaining) = remaining.split_at(parent_count);
            layer = next_layer;
            remaining = next_remaining;
        }
    }

    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &Commitment,
        indices: &[usize],
        leaf_hashes: &[Hash],
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        // Validate indices.
        verify!(indices.len() == leaf_hashes.len());
        verify!(indices.iter().all(|&i| i < self.num_leaves));
        if indices.is_empty() {
            return Ok(());
        }

        // Sort indices and leaf hashes.
        let mut layer = zip_strict(indices.iter().copied(), leaf_hashes.iter().copied())
            .collect::<Vec<(usize, Hash)>>();
        layer.sort_unstable_by_key(|(i, _)| *i);

        // Check duplicate leaf consistency and deduplicate.
        for i in 1..layer.len() {
            if layer[i - 1].0 == layer[i].0 {
                verify!(layer[i - 1].1 == layer[i].1);
            }
        }
        layer.dedup_by_key(|(i, _)| *i);

        // Validate the layers bottom-to-top.
        let mut indices = layer.iter().map(|(i, _)| *i).collect::<Vec<_>>();
        let mut hashes = layer.iter().map(|(_, h)| *h).collect::<Vec<_>>();
        let mut next_indices = Vec::with_capacity(layer.len());
        let mut input_hashes = Vec::with_capacity(layer.len() * MAX_ARITY);
        let mut next_hashes = Vec::with_capacity(layer.len());
        for layer_config in self.layers.iter().rev() {
            let arity = layer_config.arity;
            next_indices.clear();
            input_hashes.clear();
            next_hashes.clear();

            // Group hashes by parent, filling missing siblings from hints.
            let mut indices_iter = indices.iter().copied().peekable();
            let mut hashes_iter = hashes.iter().copied();
            loop {
                match indices_iter.next() {
                    Some(a) => {
                        let parent = a / arity;
                        let group_start = parent * arity;

                        // Collect known hashes for this sibling group.
                        let mut group = [None; MAX_ARITY];
                        group[a - group_start] = Some(hashes_iter.next().unwrap());
                        while let Some(&b) = indices_iter.peek() {
                            if b / arity == parent {
                                group[b - group_start] = Some(hashes_iter.next().unwrap());
                                indices_iter.next();
                            } else {
                                break;
                            }
                        }

                        // Push all `arity` children in order, reading hints for
                        // missing positions.
                        for slot in &group[..arity] {
                            match slot {
                                Some(h) => input_hashes.push(*h),
                                None => input_hashes.push(verifier_state.prover_hint()?),
                            }
                        }
                        next_indices.push(parent);
                    }
                    None => break,
                }
            }

            // Compute next layer hashes
            let hash_size = node_hash_size(arity);
            next_hashes.resize(input_hashes.len() / arity, Hash::default());
            ENGINES
                .retrieve(layer_config.hash_id)
                .ok_or(VerificationError)?
                .hash_many(hash_size, input_hashes.as_bytes(), &mut next_hashes);
            swap(&mut indices, &mut next_indices);
            swap(&mut hashes, &mut next_hashes);
        }

        // We should be left with a single root hash, matching the commitment.
        verify!(indices == [0]);
        verify!(hashes == [commitment.hash]);
        Ok(())
    }
}

impl Witness {
    pub const fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Compute the optimal mixed-arity layer sequence for a tree with `size` leaves.
///
/// Returns a vector of arities (root-to-bottom) matching the NTT domain
/// decomposition of `size`.
///
/// `size` must be a smooth-{2,3,13} number (`2^a * 3^b * 13^c`). The tree
/// uses `a` binary layers on top, `b` ternary layers in the middle, and `c`
/// arity-13 layers on the bottom. This deterministic layout lets
/// circuit-based verifiers (e.g. Gnark) hardcode the arity sequence at
/// compile time.
///
/// # Panics
///
/// Panics if `size` is not a smooth-{2,3,13} number.
pub fn layers_for_size(size: usize) -> Vec<usize> {
    if size <= 1 {
        return Vec::new();
    }

    let (a, b, c) = crate::smooth_domain::decompose(size);

    // Root-to-bottom: binary layers first, then ternary, then arity-13.
    let mut arities = Vec::with_capacity(a + b + c);
    for _ in 0..a {
        arities.push(2);
    }
    for _ in 0..b {
        arities.push(3);
    }
    for _ in 0..c {
        arities.push(13);
    }
    arities
}

/// Maximum node hash size across all supported arities (arity 3 → 96 bytes).
/// Used for hash engine compatibility checks.
pub const MAX_NODE_HASH_SIZE: usize = node_hash_size(MAX_ARITY);

#[cfg(not(feature = "parallel"))]
fn parallel_hash(engine: &dyn HashEngine, size: usize, input: &[u8], output: &mut [Hash]) {
    engine.hash_many(size, input, output);
}

#[cfg(feature = "parallel")]
fn parallel_hash(engine: &dyn HashEngine, size: usize, input: &[u8], output: &mut [Hash]) {
    use crate::utils::workload_size;
    assert_eq!(input.len(), size * output.len());
    if input.len() > workload_size::<u8>() && output.len() >= 2 {
        // Split on output count so input bytes stay aligned to `size`.
        let split = output.len() / 2;
        let (input_a, input_b) = input.split_at(split * size);
        let (output_a, output_b) = output.split_at_mut(split);
        rayon::join(
            || parallel_hash(engine, size, input_a, output_a),
            || parallel_hash(engine, size, input_b, output_b),
        );
    } else {
        engine.hash_many(size, input, output);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use proptest::{collection::vec, prelude::Strategy};

    use super::*;
    use crate::{
        hash::{tests::hash_for_size, BLAKE3},
        transcript::{codecs::Empty, DomainSeparator},
    };

    pub fn config(num_leaves: usize) -> impl Strategy<Value = Config> {
        let base_arities = layers_for_size(num_leaves);
        let min_layers = base_arities.len();
        // Add up to three unnecessary extra binary layers at the front.
        let num_extra = 0..=3usize;
        num_extra.prop_flat_map(move |extra| {
            // Arities: extra binary layers at the front, then base arities.
            let mut arities = Vec::with_capacity(min_layers + extra);
            for _ in 0..extra {
                arities.push(2usize);
            }
            arities.extend(&base_arities);

            // For each layer, pick a hash function that supports its node_hash_size.
            // Use MAX_NODE_HASH_SIZE so we can use any hash engine for all layers.
            let layer_count = arities.len();
            let arities_for_map = arities.clone();
            vec(hash_for_size(MAX_NODE_HASH_SIZE), layer_count..=layer_count).prop_map(
                move |hash_ids| {
                    let layers = hash_ids
                        .into_iter()
                        .zip(arities_for_map.iter())
                        .map(|(hash_id, &arity)| LayerConfig { hash_id, arity })
                        .collect();
                    Config { num_leaves, layers }
                },
            )
        })
    }

    fn run_merkle_tree_test(num_leaves: usize, arities: &[usize], indices: &[usize]) {
        crate::tests::init();
        let layers: Vec<LayerConfig> = arities
            .iter()
            .map(|&arity| LayerConfig {
                hash_id: BLAKE3,
                arity,
            })
            .collect();
        let config = Config { num_leaves, layers };

        let leaves = (0..num_leaves)
            .map(|i| Hash([i as u8; 32]))
            .collect::<Vec<_>>();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let tree = config.commit(&mut prover_state, leaves);
        config.open(&mut prover_state, &tree, indices);
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let root = config.receive_commitment(&mut verifier_state).unwrap();
        let leaf_hashes: Vec<Hash> = indices.iter().map(|&i| Hash([i as u8; 32])).collect();
        config
            .verify(&mut verifier_state, &root, indices, &leaf_hashes)
            .unwrap();
    }

    #[test]
    fn test_merkle_tree_binary() {
        // Pure binary tree: 2^3 = 8 leaves.
        run_merkle_tree_test(8, &[2, 2, 2], &[1, 5, 7]);
    }

    #[test]
    fn test_merkle_tree_ternary() {
        // Pure ternary tree: 3^3 = 27 leaves.
        run_merkle_tree_test(27, &[3, 3, 3], &[5, 13, 26]);
    }

    #[test]
    fn test_merkle_tree_mixed_2a_3b() {
        // Mixed: 2^2 * 3^1 = 12 leaves. Binary top, ternary bottom.
        run_merkle_tree_test(12, &[2, 2, 3], &[0, 5, 11]);
    }

    #[test]
    fn test_merkle_tree_mixed_large() {
        // Mixed: 2^3 * 3^2 = 72 leaves.
        run_merkle_tree_test(72, &[2, 2, 2, 3, 3], &[0, 13, 42, 71]);
    }

    #[test]
    fn test_merkle_tree_arity_13() {
        // Pure arity-13: 13 leaves.
        run_merkle_tree_test(13, &[13], &[0, 6, 12]);
        // Mixed: 2 * 13 = 26 leaves.
        run_merkle_tree_test(26, &[2, 13], &[0, 12, 25]);
        // Mixed: 2 * 3 * 13 = 78 leaves.
        run_merkle_tree_test(78, &[2, 3, 13], &[0, 39, 77]);
    }

    #[test]
    fn test_merkle_tree_smooth_domain_sizes() {
        // Verify trees work for all smooth sizes with exact capacity.
        for &size in &[
            2, 3, 4, 6, 8, 9, 12, 13, 16, 18, 24, 26, 36, 39, 48, 72, 78, 104, 117,
        ] {
            let arities = layers_for_size(size);
            assert_eq!(arities.iter().product::<usize>(), size);
            run_merkle_tree_test(size, &arities, &[0, size / 2, size - 1]);
        }
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        run_merkle_tree_test(1, &[], &[0]);
    }

    #[test]
    fn test_layers_for_size() {
        // Edge cases.
        let empty: Vec<usize> = vec![];
        assert_eq!(layers_for_size(0), empty);
        assert_eq!(layers_for_size(1), empty);

        // Pure powers of 2.
        assert_eq!(layers_for_size(2), vec![2]);
        assert_eq!(layers_for_size(4), vec![2, 2]);
        assert_eq!(layers_for_size(8), vec![2, 2, 2]);
        assert_eq!(layers_for_size(16), vec![2, 2, 2, 2]);

        // Pure powers of 3.
        assert_eq!(layers_for_size(3), vec![3]);
        assert_eq!(layers_for_size(9), vec![3, 3]);

        // Mixed: binary on top, ternary on bottom.
        assert_eq!(layers_for_size(6), vec![2, 3]); // 2 * 3
        assert_eq!(layers_for_size(12), vec![2, 2, 3]); // 4 * 3
        assert_eq!(layers_for_size(18), vec![2, 3, 3]); // 2 * 9
        assert_eq!(layers_for_size(36), vec![2, 2, 3, 3]); // 4 * 9
        assert_eq!(layers_for_size(72), vec![2, 2, 2, 3, 3]); // 8 * 9

        // Arity-13 sizes.
        assert_eq!(layers_for_size(13), vec![13]); // 13
        assert_eq!(layers_for_size(26), vec![2, 13]); // 2 * 13
        assert_eq!(layers_for_size(39), vec![3, 13]); // 3 * 13
        assert_eq!(layers_for_size(78), vec![2, 3, 13]); // 2 * 3 * 13
        assert_eq!(layers_for_size(117), vec![3, 3, 13]); // 9 * 13
        assert_eq!(layers_for_size(104), vec![2, 2, 2, 13]); // 8 * 13

        // Capacity equals size exactly (no over-provisioning).
        for &size in &[
            2, 3, 4, 6, 8, 9, 12, 13, 16, 18, 24, 26, 36, 39, 48, 72, 78, 96, 104, 117, 144,
        ] {
            let arities = layers_for_size(size);
            let cap: usize = arities.iter().product();
            assert_eq!(cap, size, "size={size}, cap={cap}, arities={arities:?}");
        }
    }

    #[test]
    #[should_panic(expected = "not a smooth-{2,3,13} number")]
    fn test_layers_for_size_rejects_non_smooth() {
        layers_for_size(5);
    }

    #[test]
    fn test_num_nodes() {
        // 0 layers: just the root = 1 node
        let c0 = Config {
            num_leaves: 1,
            layers: vec![],
        };
        assert_eq!(c0.num_nodes(), 1);

        // Binary: 2 layers → 4 leaves + 2 internal + 1 root = 7
        let c_bin = Config {
            num_leaves: 4,
            layers: vec![
                LayerConfig {
                    hash_id: BLAKE3,
                    arity: 2,
                },
                LayerConfig {
                    hash_id: BLAKE3,
                    arity: 2,
                },
            ],
        };
        assert_eq!(c_bin.num_nodes(), 7);

        // Ternary: 1 layer → 3 leaves + 1 root = 4
        let c_ter = Config {
            num_leaves: 3,
            layers: vec![LayerConfig {
                hash_id: BLAKE3,
                arity: 3,
            }],
        };
        assert_eq!(c_ter.num_nodes(), 4);

        // Mixed: [2, 3] → leaf capacity = 6. Nodes: 6 + 2 + 1 = 9
        let c_mix = Config {
            num_leaves: 6,
            layers: vec![
                LayerConfig {
                    hash_id: BLAKE3,
                    arity: 2,
                },
                LayerConfig {
                    hash_id: BLAKE3,
                    arity: 3,
                },
            ],
        };
        assert_eq!(c_mix.num_nodes(), 9);
    }

    #[test]
    fn test_merkle_tree_243_ternary() {
        // 243 = 3^5, all ternary like the original test.
        run_merkle_tree_test(243, &[3, 3, 3, 3, 3], &[13, 42]);
    }

    #[test]
    fn test_merkle_tree_256_mixed() {
        // 256 = 2^8, all binary.
        run_merkle_tree_test(256, &[2, 2, 2, 2, 2, 2, 2, 2], &[13, 42, 255]);
    }
}
