//! Merkle Tree implementation.

mod digest;

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use spongefish::{
    codecs::{
        arkworks::{ArkworksHintPattern, ArkworksHintProver, ArkworksHintVerifier},
        ZeroCopyHintProver, ZeroCopyHintVerifier, ZeroCopyPattern, ZeroCopyProver,
        ZeroCopyVerifier,
    },
    transcript::{Label, Length},
    Unit,
};
use thiserror::Error;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub use self::digest::{ArkDigestEngine, DigestEngine};

pub type Hash = [u8; 32];

pub trait Engine<T>: Sync + Send + Debug + Display {
    /// Hash many `leaf_size` values of type `T` to a 256-bit outputs.
    ///
    /// # Input contract
    /// ```
    /// assert_eq!(out.len() * leaf_size, input.len());
    /// ```
    fn leaf_hash(&self, input: &[T], leaf_size: usize, out: &mut [Hash]);

    /// Hash many pairs of 256-bit messages to 256-bit outputs.
    ///
    /// # Input contract
    /// ```
    /// assert_eq!(2 * out.len(), input.len());
    /// ```
    fn node_hash(&self, input: &[Hash], out: &mut [Hash]);
}

/// Merkle Tree configuration for a tree with `num_leaves` leaves, each containing `leaf_size` values of type `T`.
#[derive(Debug, Clone)]
pub struct Config<T> {
    engine: Arc<dyn Engine<T>>,
    leaf_size: usize,
    num_leaves: usize,
    // TODO: blinding: bool,
    // TODO: merkle_cap: usize,
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

pub trait MerkleTreePattern<U>: ZeroCopyPattern<U> + ArkworksHintPattern<U>
where
    U: Unit,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
    ) -> Result<(), Self::Error>;

    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error>;

    fn merkle_tree_open_with_leaves<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes;

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize;
}

pub trait MerkleTreeProver<U>: ZeroCopyProver<U> + ArkworksHintProver<U>
where
    U: Unit,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        values: &[T],
    ) -> Result<Witness, Self::Error>;

    /// Open without hinting leaves
    /// Useful to save proof size when verifier has them by other means.
    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>;

    fn merkle_tree_open_with_leaves<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        values: &[T],
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes;

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        values: &[T],
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize;
}

pub trait MerkleTreeVerifier<'a, U>: ZeroCopyVerifier<'a, U> + ArkworksHintVerifier<'a, U>
where
    U: Unit,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
    ) -> Result<Commitment, Self::Error>;

    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
        leaves: &[T],
    ) -> Result<(), VerifierError<Self::Error>>;

    fn merkle_tree_open_with_leaves_out<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
        out: &mut [T],
    ) -> Result<(), VerifierError<Self::Error>>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes;

    fn merkle_tree_open_with_leaves_vec<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
    ) -> Result<Vec<T>, VerifierError<Self::Error>>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes,
    {
        let mut result =
            T::new_vec_zeroed(indices.len() * config.leaf_size).expect("Allocation error");
        self.merkle_tree_open_with_leaves_out(label, config, commitment, indices, &mut result)?;
        Ok(result)
    }

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
    ) -> Result<Vec<T>, VerifierError<Self::Error>>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize;
}

impl<T> Config<T> {
    pub fn new(engine: Arc<dyn Engine<T>>, leaf_size: usize, num_leaves: usize) -> Self {
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

    pub fn num_nodes(&self) -> usize {
        // The number of nodes is num_leaves - 1.
        2 * self.num_leaves - 1
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
    P: ZeroCopyPattern<U> + ArkworksHintPattern<U>,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))?;
        self.message_zerocopy::<Commitment>("root")?;
        self.end_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))
    }

    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        _config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error> {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))?;
        self.hint_zerocopies_dynamic::<Hash>("merkle-proof")?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))
    }

    fn merkle_tree_open_with_leaves<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))?;
        self.hint_zerocopies::<T>("opened-leaves", config.leaf_size * num_open)?;
        self.merkle_tree_open("merkle-proof", config, num_open)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))
    }

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        num_open: usize,
    ) -> Result<(), Self::Error>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))?;
        self.hint_arkworks::<Vec<T>>("opened-leaves")?;
        self.merkle_tree_open("merkle-proof", config, num_open)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(num_open))
    }
}

impl<U, P> MerkleTreeProver<U> for P
where
    U: Unit,
    P: ZeroCopyProver<U> + ZeroCopyHintProver<U> + ArkworksHintProver<U>,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        leaves: &[T],
    ) -> Result<Witness, Self::Error> {
        assert_eq!(leaves.len(), config.leaf_size * config.num_leaves);
        let label = label.into();
        self.begin_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))?;

        // Allocate nodes and fill with leaf layer.
        let mut nodes = vec![Hash::default(); config.num_nodes()];
        config
            .engine
            .leaf_hash(leaves, config.leaf_size, &mut nodes[..config.num_leaves]);

        // Compute merkle tree nodes.
        let (mut previous, mut remaining) = nodes.split_at_mut(config.num_leaves);
        while !remaining.is_empty() {
            let (current, next_remaining) = remaining.split_at_mut(previous.len() / 2);
            config.engine.node_hash(previous, current);
            previous = current;
            remaining = next_remaining;
        }

        // Create witness
        let witness: Witness = Witness {
            commitment: Commitment {
                root: *nodes.last().unwrap(),
            },
            nodes,
        };
        self.message_zerocopy::<Commitment>("root", &witness.commitment)?;
        self.end_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))?;
        Ok(witness)
    }

    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error> {
        let label = label.into();
        let size = indices.len();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(size))?;
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
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(size))
    }

    fn merkle_tree_open_with_leaves<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        values: &[T],
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        let mut opened_leaves = Vec::with_capacity(indices.len() * config.leaf_size);
        for index in indices {
            let leaf = &values[index * config.leaf_size..(index + 1) * config.leaf_size];
            opened_leaves.extend_from_slice(leaf);
        }
        self.hint_zerocopy_slice("opened-leaves", &opened_leaves)?;
        self.merkle_tree_open("merkle-proof", config, witness, indices)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))
    }

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        values: &[T],
        witness: &Witness,
        indices: &[usize],
    ) -> Result<(), Self::Error>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        let mut opened_leaves = Vec::with_capacity(indices.len() * config.leaf_size);
        for index in indices {
            let leaf = &values[index * config.leaf_size..(index + 1) * config.leaf_size];
            opened_leaves.extend_from_slice(leaf);
        }
        self.hint_arkworks::<Vec<T>>("opened-leaves", &opened_leaves)
            .expect("TODO"); // TODO
        self.merkle_tree_open("merkle-proof", config, witness, indices)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))
    }
}

impl<'a, U, P> MerkleTreeVerifier<'a, U> for P
where
    U: Unit,
    P: ZeroCopyVerifier<'a, U> + ZeroCopyHintVerifier<'a, U> + ArkworksHintVerifier<'a, U>,
{
    fn merkle_tree_commit<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
    ) -> Result<Commitment, Self::Error> {
        let label = label.into();
        self.begin_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))?;
        let commitment = self.message_zerocopy::<Commitment>("root")?;
        self.end_message::<Config<T>>(label.clone(), Length::Fixed(config.num_leaves))?;
        Ok(commitment)
    }

    fn merkle_tree_open<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
        leaves: &[T],
    ) -> Result<(), VerifierError<Self::Error>> {
        let label = label.into();
        let size = indices.len();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(size))?;
        if !indices.iter().all(|&i| i < config.num_leaves) {
            return Err(VerifierError::OutOfBounds);
        }

        // Compute leaf hashes.
        assert_eq!(leaves.len() % config.leaf_size, 0);
        let mut leaf_hashes = vec![Hash::default(); leaves.len() / config.leaf_size];
        config
            .engine
            .leaf_hash(leaves, config.leaf_size, &mut leaf_hashes);

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
            config.engine.node_hash(&input_hashes, &mut hashes);
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

        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(size))?;
        Ok(())
    }

    fn merkle_tree_open_with_leaves_out<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
        out: &mut [T],
    ) -> Result<(), VerifierError<Self::Error>>
    where
        T: Clone + Immutable + KnownLayout + IntoBytes + FromBytes,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        assert_eq!(out.len(), indices.len() * config.leaf_size);
        self.hint_zerocopy_slice_out::<T>("opened-leaves", out)?;
        dbg!();
        self.merkle_tree_open("merkle-proof", config, commitment, indices, out)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        Ok(())
    }

    fn merkle_tree_open_with_leaves_ark<T>(
        &mut self,
        label: impl Into<Label>,
        config: &Config<T>,
        commitment: &Commitment,
        indices: &[usize],
    ) -> Result<Vec<T>, VerifierError<Self::Error>>
    where
        T: Clone + CanonicalSerialize + CanonicalDeserialize,
    {
        let label = label.into();
        self.begin_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        let result = self.hint_arkworks::<Vec<T>>("opened-leaves").expect("TODO");
        assert_eq!(result.len(), indices.len() * config.leaf_size);
        self.merkle_tree_open("merkle-proof", config, commitment, indices, &result)?;
        self.end_hint::<Config<T>>(label.clone(), Length::Fixed(indices.len()))?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use sha3::Keccak256;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_all_ops() -> Result<()> {
        let config = Config {
            engine: Arc::new(DigestEngine::<Keccak256>::default()),
            leaf_size: 3,
            num_leaves: 64,
        };

        let leaves = (0_usize..)
            .map(|i| i as u16)
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
        pattern.merkle_tree_open_with_leaves("open", &config, 5)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        let witness = prover.merkle_tree_commit("commit", &config, &leaves)?;
        prover.merkle_tree_open_with_leaves("open", &config, &leaves, &witness, &indices)?;
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "5061cc0d60d0544547199b9c9b48abac8261edb3af541a2b9fc28404f3c3cf140f00100011000600070008000c000d000e00180019001a00180019001a0000010000322e1eadc3e16bb67c3619059313509ce0281a29ab6c597fa43dcfed9a7c53e94db7ae4d009325d260cced49bbf0c094a31e56edc58b70b0c75a32b840cc77ddae1a26f31e0ffc20fd88186342959c25bfe5627e719578613f2b6aa2bfea715370cc56e80ce72271e7ed64de9368e4f2e680537f5222078a7e3c10c9ffba5d69f68803e322b0f7653160a2c40627babf3faa442bb6a550e699e37af35d64407dac591d8e5787450ac5557a7fc0d467e568ef1a467b037251e4dd3a6b4cf10a39c5e45c796b2d7b963fe52f05f2c453f91b190f64daff2976b643f9ff622c3f4ec43db12a16ad00a5c50deff1d9db1b440b02107a2e29268d80b4fa79991f3456");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        let commitment = verifier.merkle_tree_commit("commit", &config)?;
        assert_eq!(commitment.root, witness.commitment.root);
        let leaves =
            verifier.merkle_tree_open_with_leaves_vec("open", &config, &commitment, &indices)?;
        assert_eq!(leaves, opening);
        verifier.finalize()?;

        Ok(())
    }

    #[test]
    fn test_all_ops_ark() -> Result<()> {
        let config = Config {
            engine: Arc::new(ArkDigestEngine::<Keccak256>::default()),
            leaf_size: 3,
            num_leaves: 64,
        };

        let leaves = (0_usize..)
            .map(|i| Field64::from(i as u64))
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
        pattern.merkle_tree_open_with_leaves_ark("open", &config, 5)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        let witness = prover.merkle_tree_commit("commit", &config, &leaves)?;
        prover.merkle_tree_open_with_leaves_ark("open", &config, &leaves, &witness, &indices)?;
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "ce517044d968426f8b1605e8edde29c0aba02f55e8595367307b56acf235f06d800000000f000000000000000f00000000000000100000000000000011000000000000000600000000000000070000000000000008000000000000000c000000000000000d000000000000000e00000000000000180000000000000019000000000000001a00000000000000180000000000000019000000000000001a000000000000000001000024e201b1f6e80fd3df6596ac3d1a479429d71a593b68894eee55dec7c7dfc7106a59cd3c8c11c1ef150546d3721909f1ea278b75f78c4c303d08af355327561d385af81684516f59d756b2377b3c87d519cc2308502b3f9d7af214e41a5e134a8a25d7576aaeb3a302e4589b1f9bff25bbe4870e3403655e66e4f4a17c00f756c750c0330585823c57f2631548e1683b0ca665bdd16b172c119caf463f3ac43cafd86797c4b20ff8bd8975ccdc480a5ad5125cac7af8203ebf4206ac4930bdf3ededc7442672ce099d0ba4197bfbc6a6614324ffc5d2d0de6b77485c732c968bce7be01bb41403fe2531c286580c5d727d2ba8d204558c588dcd82d55c8a2f45");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        let commitment = verifier.merkle_tree_commit("commit", &config)?;
        assert_eq!(commitment.root, witness.commitment.root);
        let leaves =
            verifier.merkle_tree_open_with_leaves_ark("open", &config, &commitment, &indices)?;
        assert_eq!(leaves, opening);
        verifier.finalize()?;

        Ok(())
    }
}
