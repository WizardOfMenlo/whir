mod digest;

use std::{
    fmt::{Debug, Display},
    mem::swap,
    sync::Arc,
};

use rand::seq::index;
use spongefish::{
    codecs::{
        ZeroCopyHintProver, ZeroCopyHintVerifier, ZeroCopyPattern, ZeroCopyProver, ZeroCopyVerifier,
    },
    transcript::{Label, Length},
    Unit,
};
use thiserror::Error;
use zerocopy::{transmute_ref, FromBytes, Immutable, IntoBytes, KnownLayout};

pub use self::digest::DigestEngine;

pub type Hash = [u8; 32];

pub trait Engine: Sync + Send + Debug + Display {
    /// Hash many 512-bit messages to 256-bit outputs.
    fn hash_many(&self, input: &[Hash], out: &mut [Hash]);
}

#[derive(Debug, Clone)]
pub struct Config {
    engine: Arc<dyn Engine>,
    leaf_size: usize,
    num_leaves: usize,
}

pub struct Witness {
    commitment: Commitment,
    // The nodes in the Merkle tree, starting with the leaf hash layer.
    nodes: Vec<Hash>,
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Immutable,
    KnownLayout,
    FromBytes,
    IntoBytes,
)]
#[repr(transparent)]
pub struct Commitment {
    root: Hash,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Error)]
pub enum VerifierError<E> {
    #[error(transparent)]
    Inner(#[from] E),
    #[error("Leaf index was out of bounds.")]
    OutOfBounds,
    #[error("Merkle tree root did not match.")]
    RootMismatch,
    #[error("A duplicated leaf was inconsistent.")]
    DuplicateLeafMismatch,
}

pub trait MerkleTreePattern<U>: ZeroCopyPattern<U>
where
    U: Unit,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), Self::Error>;

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        num_open: usize,
    ) -> Result<(), Self::Error>;
}

pub trait MerkleTreeProver<U>: ZeroCopyProver<U>
where
    U: Unit,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        leaves: &[u8],
    ) -> Result<Witness, Self::Error>;

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>;
}

pub trait MerkleTreeVerifier<'a, U>: ZeroCopyVerifier<'a, U>
where
    U: Unit,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<Commitment, Self::Error>;

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        commitment: &Commitment,
        indices: &[usize],
        leaves: &[u8],
    ) -> Result<(), VerifierError<Self::Error>>;
}

impl Config {
    pub fn new(engine: Arc<dyn Engine>, leaf_size: usize, num_leaves: usize) -> Self {
        // TODO: Solve restrictions by padding and by carrying final left element directly to next layer.
        // leaf size must be a power-of-two multiple of 32 bytes.
        assert!(
            leaf_size.is_power_of_two(),
            "Leaf size must be a power of two"
        );
        assert!(
            leaf_size.trailing_zeros() >= 6,
            "Leaf size must be a multiple of 64 bytes"
        );
        // num_leaves must be a power of two.
        assert!(
            num_leaves.is_power_of_two(),
            "Number of leaves must be a power of two"
        );
        Self {
            engine,
            leaf_size,
            num_leaves,
        }
    }

    pub fn num_layers(&self) -> usize {
        // The number of layers is log2(num_leaves).
        self.num_leaves.trailing_zeros() as usize
    }

    pub fn num_nodes(self) -> usize {
        // The number of nodes is num_leaves - 1.
        self.num_leaves - 1
    }
}

impl Witness {
    fn num_leaves(&self) -> usize {
        (self.nodes.len() + 1) / 2
    }

    fn layers(&self) -> impl Iterator<Item = &[Hash]> {
        let mut offset = 0;
        let mut size = self.num_leaves();
        std::iter::from_fn(move || {
            if size == 0 {
                None
            } else {
                let layer = &self.nodes[offset..offset + size];
                offset += size;
                size /= 2;
                Some(layer)
            }
        })
    }
}

impl<U, P> MerkleTreePattern<U> for P
where
    U: Unit,
    P: ZeroCopyPattern<U>,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))?;
        self.message_zerocopy::<Commitment>("root")?;
        self.end_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))
    }

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        num_open: usize,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_hint::<Config>(label.clone(), Length::Fixed(num_open))?;
        self.hint_zerocopies_dynamic::<Hash>("merkle-proof")?;
        self.end_hint::<Config>(label.clone(), Length::Fixed(num_open))
    }
}

impl<U, P> MerkleTreeProver<U> for P
where
    U: Unit,
    P: ZeroCopyProver<U> + ZeroCopyHintProver<U>,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        leaves: &[u8],
    ) -> Result<Witness, Self::Error> {
        assert_eq!(leaves.len(), config.leaf_size * config.num_leaves);
        let label = label.into();
        self.begin_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))?;

        // Compute leaf hashes by hashing pairs of leaves together.
        let leaves =
            <[Hash]>::ref_from_bytes(leaves).expect("leaf_size is multiple of (double) Hash size");
        let mut leaf_hashes = vec![Hash::default(); leaves.len() / 2];
        config.engine.hash_many(leaves, &mut leaf_hashes);
        while leaf_hashes.len() > config.num_leaves {
            let mut next_leaf_hashes = vec![Hash::default(); leaf_hashes.len() / 2];
            config.engine.hash_many(&leaf_hashes, &mut next_leaf_hashes);
            swap(&mut leaf_hashes, &mut next_leaf_hashes);
        }
        debug_assert_eq!(leaf_hashes.len(), config.num_leaves);

        // Compute merkle tree nodes.
        let mut nodes = leaf_hashes;
        nodes.resize(2 * config.num_leaves - 1, Hash::default());
        let (mut leaves, mut tail) = nodes.split_at_mut(config.num_leaves);
        while leaves.len() > 1 {
            let (next_leaves, next_tail) = tail.split_at_mut(leaves.len() / 2);
            config.engine.hash_many(leaves, next_leaves);
            leaves = next_leaves;
            tail = next_tail;
        }
        let witness: Witness = Witness {
            commitment: Commitment {
                root: *nodes.last().unwrap(),
            },
            nodes,
        };
        self.message_zerocopy::<Commitment>("root", &witness.commitment)?;
        self.end_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))?;
        Ok(witness)
    }

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error> {
        let label = label.into();
        let size = indices.len();
        self.begin_hint::<Config>(label.clone(), Length::Fixed(size))?;
        assert!(indices.iter().all(|&i| i < config.num_leaves));

        // Abstract execution of verify algorithm returning a stack of required hashes.
        let mut proof = Vec::new();
        let mut indices = indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        for layer in witness.layers().take_while(|l| l.len() > 1) {
            let mut next_indices = Vec::with_capacity(indices.len());
            let mut iter = indices.iter().copied().peekable();
            loop {
                match (iter.next(), iter.peek()) {
                    (Some(a), Some(&b)) if b == a ^ 1 => {
                        // Neighboring indices, merging branches.
                        next_indices.push(a >> 1);
                        iter.next(); // Skip the next index.
                    }
                    (Some(a), _) => {
                        // Single index, pushing the neighbor hash.
                        proof.push(layer[a ^ 1]);
                        next_indices.push(a >> 1);
                    }
                    (None, _) => break,
                }
            }
            indices = next_indices;
        }

        self.hint_zerocopy_dynamic::<Hash>("merkle-proof", &proof)?;
        self.end_hint::<Config>(label.clone(), Length::Fixed(size))
    }
}

impl<'a, U, P> MerkleTreeVerifier<'a, U> for P
where
    U: Unit,
    P: ZeroCopyVerifier<'a, U> + ZeroCopyHintVerifier<'a, U>,
{
    fn merkle_tree_commit(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
    ) -> Result<Commitment, Self::Error> {
        let label = label.into();
        self.begin_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))?;
        let commitment = self.message_zerocopy::<Commitment>("root")?;
        self.end_message::<Config>(label.clone(), Length::Fixed(config.num_leaves))?;
        Ok(commitment)
    }

    fn merkle_tree_open(
        &mut self,
        label: impl Into<Label>,
        config: &Config,
        commitment: &Commitment,
        indices: &[usize],
        leaves: &[u8],
    ) -> Result<(), VerifierError<Self::Error>> {
        let label = label.into();
        let size = indices.len();
        self.begin_hint::<Config>(label.clone(), Length::Fixed(size))?;
        if !indices.iter().all(|&i| i < config.num_leaves) {
            return Err(VerifierError::OutOfBounds);
        }

        // Compute leaf hashes by hashing pairs of leaves together.
        let leaves =
            <[Hash]>::ref_from_bytes(leaves).expect("leaf_size is multiple of (double) Hash size");
        let mut leaf_hashes = vec![Hash::default(); leaves.len() / 2];
        config.engine.hash_many(leaves, &mut leaf_hashes);
        while leaf_hashes.len() > config.num_leaves {
            let mut next_leaf_hashes = vec![Hash::default(); leaf_hashes.len() / 2];
            config.engine.hash_many(&leaf_hashes, &mut next_leaf_hashes);
            swap(&mut leaf_hashes, &mut next_leaf_hashes);
        }
        assert_eq!(leaf_hashes.len(), indices.len());

        // Sort indices and leaf hashes.
        let mut layer = indices
            .iter()
            .copied()
            .zip(leaf_hashes.into_iter())
            .collect::<Vec<(usize, Hash)>>();
        layer.sort_unstable_by_key(|(i, _)| *i);
        // Check duplicate leaf hash consistency.
        for i in 1..layer.len() {
            if layer[i - 1].0 == layer[i].0 {
                if layer[i - 1].1 != layer[i].1 {
                    return Err(VerifierError::DuplicateLeafMismatch);
                }
            }
        }
        layer.dedup_by_key(|(i, _)| *i);
        let mut indices = layer.iter().map(|(i, _)| *i).collect::<Vec<_>>();
        let mut hashes = layer.iter().map(|(_, h)| *h).collect::<Vec<_>>();

        let proof = self.hint_zerocopy_dynamic_vec::<Hash>("merkle-proof")?;
        let mut proof = proof.iter().copied();

        for _ in 0..config.num_layers() {
            debug_assert_eq!(hashes.len(), indices.len());
            // OPT: Re-use buffers more
            let mut next_indices = Vec::with_capacity(layer.len());
            let mut input_hashes = Vec::new();
            let mut indices_iter = indices.iter().copied().peekable();
            let mut hashes_iter = hashes.iter().copied();
            loop {
                match (indices_iter.next(), indices_iter.peek()) {
                    (Some(a), Some(&b)) if b == a ^ 1 => {
                        // Neighboring indices, merging branches.
                        input_hashes.push(hashes_iter.next().unwrap());
                        input_hashes.push(hashes_iter.next().unwrap());
                        next_indices.push(a >> 1);
                        indices_iter.next(); // Skip the next index.
                    }
                    (Some(a), _) => {
                        // Single index, pushing the neighbor hash.
                        if a & 1 == 0 {
                            input_hashes.push(hashes_iter.next().unwrap());
                            input_hashes.push(proof.next().unwrap());
                        } else {
                            input_hashes.push(proof.next().unwrap());
                            input_hashes.push(hashes_iter.next().unwrap());
                        }
                        next_indices.push(a >> 1);
                    }
                    (None, _) => break,
                }
            }
            hashes.truncate(input_hashes.len() / 2);
            // Compute next layer hashes in a single batch for efficiency.
            config.engine.hash_many(&input_hashes, &mut hashes);
            indices = next_indices;
        }
        // TODO: Some of these should probably be errors.
        assert_eq!(proof.next(), None);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_eq!(hashes.len(), 1);

        if hashes[0] != commitment.root {
            self.abort(); // TODO: Better devex so you can't forget this.
            return Err(VerifierError::RootMismatch);
        }

        self.end_hint::<Config>(label.clone(), Length::Fixed(size))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use sha3::Keccak256;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config {
            engine: Arc::new(DigestEngine::<Keccak256>::default()),
            leaf_size: 64,
            num_leaves: 16,
        };

        let leaves = (0_usize..)
            .map(|i| i as u8)
            .take(config.leaf_size * config.num_leaves)
            .collect::<Vec<_>>();

        let indices = [5, 2, 4, 8, 8];

        let opening = indices
            .iter()
            .flat_map(|&i| {
                leaves[i * config.leaf_size..(i + 1) * config.leaf_size]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();

        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.merkle_tree_commit("commit", &config)?;
        pattern.merkle_tree_open("open", &config, 5)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        let witness = prover.merkle_tree_commit("commit", &config, &leaves)?;
        prover.merkle_tree_open("open", &config, &witness, &indices)?;
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "bc6fe41a519a46afbac1e0b61c03ed41b46ba45abba7c3e1b5503534ef46c2b0c0000000db13df11c88c146de5d0fc8e57b83142e8ad32c6e6495c034a7e0cb1b5b6ac7be7519b8a28eebd70a39cf1ff0dd7a2c3c4d2125a61a98301a4b46d88eec1ffe3a28b94ad26dbb5acac572480b4b07461ad6fccdb82598e71f949b14be96c5050df8d05daec2710dcd518728eb1750b2c5609bc915fc29a2430e5590c24a4a5bbdf8d05daec2710dcd518728eb1750b2c5609bc915fc29a2430e5590c24a4a5bbaf7761c7e694cf080ed33423d7799c01e81bae30c1d61233417751dea61a708d");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        let commitment = verifier.merkle_tree_commit("commit", &config)?;
        assert_eq!(commitment.root, witness.commitment.root);
        dbg!(&commitment);
        verifier.merkle_tree_open("open", &config, &commitment, &indices, &opening)?;
        verifier.finalize()?;

        Ok(())
    }
}
