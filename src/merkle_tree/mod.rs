mod hasher;

use std::mem::swap;

pub use hasher::{Hash, Hasher, Hashers, HASH_ZERO};

pub struct MerkleTreeHasher {
    depth: usize,

    // TODO: Different hashers at different depths.
    hasher: Box<dyn Hasher>,
}

pub enum Error {
    IndexOutOfBounds,
    InsufficientSibblings,
    ExcessSibblings,
    LeafMismatch,
    RootMismatch,
}

impl MerkleTreeHasher {
    pub fn new(depth: usize, hasher: Hashers) -> Self {
        Self {
            depth,
            hasher: hasher.construct(),
        }
    }

    pub fn size_at_depth(depth: usize) -> usize {
        1 << depth
    }

    pub fn hasher_at_depth(&self, _depth: usize) -> &dyn Hasher {
        &*self.hasher
    }

    /// Verify a Merkle tree multi-proof with path merging.
    ///
    /// Can also verify a single proof by passing a single leaf and index, or a
    /// non-merging proof as repeated single leaf proofs.
    pub fn verify(
        &self,
        root: Hash,
        indices: &[usize],
        leaves: &[Hash],
        proof: &[Hash],
    ) -> Result<(), Error> {
        // Ensure indices are in-bounds
        let index_bound = Self::size_at_depth(self.depth);
        if indices.iter().any(|&i| i >= index_bound) {
            return Err(Error::IndexOutOfBounds);
        }

        // Sort leaves by indices
        let mut combined = indices
            .iter()
            .copied()
            .zip(leaves.iter().copied())
            .collect::<Vec<_>>();
        combined.sort_by_key(|&(i, _)| i);

        // Deduplicate leaves
        let mut indices = Vec::with_capacity(combined.len());
        let mut leaves = Vec::with_capacity(combined.len());
        for (i, leaf) in combined {
            if indices.last().copied() == Some(i) {
                if leaves.last() != Some(&leaf) {
                    return Err(Error::LeafMismatch);
                }
            } else {
                indices.push(i);
                leaves.push(leaf);
            }
        }

        // The proof is a list of sibblings in the order they are required
        // to merge with the leaves and compute the root.
        let mut siblings = proof.iter().copied();

        // Repeatedly compute the next layer hashes and indices.
        let mut depth = self.depth;
        let mut layer = Vec::with_capacity(2 * leaves.len());
        let mut next_indices = Vec::with_capacity(indices.len());
        let mut next_layer = Vec::new();
        while depth > 0 {
            // Complete the current layer, adding missing sibblings
            assert_eq!(indices.len(), leaves.len());
            let mut indices_it = indices.iter().copied().peekable();
            let mut leaves_it = leaves.iter().copied();
            while let Some(index) = indices_it.next() {
                let leaf = leaves_it.next().unwrap();
                if index % 2 == 0 {
                    // Left child, check if next leaf is right child else fetch a sibbling.
                    layer.push(leaf);
                    layer.push(if indices_it.peek().copied() == Some(index + 1) {
                        // It is, merge two sibblings
                        indices_it.next();
                        leaves_it.next().unwrap()
                    } else {
                        // It is not, fetch a sibbling.
                        siblings.next().ok_or(Error::InsufficientSibblings)?
                    });
                } else {
                    // Right child (and we didn't merge in last round), fetch left sibbling
                    layer.push(siblings.next().ok_or(Error::InsufficientSibblings)?);
                    layer.push(leaf);
                }
                next_indices.push(index / 2);
            }

            // Compute the next layer hashes
            assert_eq!(layer.len() % 2, 0);
            next_layer.resize(layer.len() / 2, HASH_ZERO);
            self.hasher_at_depth(depth)
                .hash_pairs(&layer, &mut next_layer);

            // Repeat, re-using vecs
            depth -= 1;
            swap(&mut indices, &mut next_indices);
            swap(&mut leaves, &mut next_layer);
            next_indices.clear();
            next_layer.clear();
        }

        // Make sure we consumed all sibblings
        if siblings.next().is_some() {
            return Err(Error::ExcessSibblings);
        }

        // Check root
        assert!(leaves.len() == 1);
        if leaves[0] != root {
            return Err(Error::RootMismatch);
        }

        Ok(())
    }
}
