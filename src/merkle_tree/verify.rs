use {
    super::*,
    std::{iter::zip, mem::swap},
};

impl MerkleTreeHasher {
    /// Verify a Merkle tree proof.
    pub fn verify_single(
        &self,
        root: &Hash,
        leaf_index: usize,
        leaf_hash: Hash,
        proof: &[Hash],
    ) -> Result<(), Error> {
        self.verify_inner(root, &[leaf_index], &[leaf_hash], proof, false)?;
        Ok(())
    }

    /// Verify a Merkle tree multi-proof with path merging.
    ///
    /// Can also verify a single proof by passing a single leaf and index, or a
    /// non-merging proof as repeated single leaf proofs.
    ///
    pub fn verify_multi(
        &self,
        root: &Hash,
        leaf_indices: &[usize],
        leaf_hashes: &[Hash],
        proof: &[Hash],
    ) -> Result<(), Error> {
        self.verify_inner(root, leaf_indices, leaf_hashes, proof, false)?;
        Ok(())
    }

    /// Expand a multi-proof to individual proofs.
    ///
    /// This can aid arithmatized verification where the control flow of
    /// multi-proof verification is hard to implement.
    pub fn expand_proof(
        &self,
        root: &Hash,
        leaf_indices: &[usize],
        leaf_hashes: &[Hash],
        proof: &[Hash],
    ) -> Result<Vec<Vec<Hash>>, Error> {
        self.verify_inner(root, leaf_indices, leaf_hashes, proof, true)
    }

    /// Verify a Merkle tree multi-proof with path merging, optionally producing
    /// single proofs.
    pub fn verify_inner(
        &self,
        root: &Hash,
        leaf_indices: &[usize],
        leaf_hashes: &[Hash],
        proof: &[Hash],
        produce_single_proofs: bool,
    ) -> Result<Vec<Vec<Hash>>, Error> {
        // Sort and deduplicate the indices and leaves.
        let (mut indices, mut leaves) = self.sort_indices(leaf_indices, leaf_hashes)?;
        // The proof is a list of siblings in the order they are required
        // to merge with the leaves and compute the root.
        let mut siblings = proof.iter().copied();
        // Allocate for single proofs if requested.
        let mut single_proofs = if produce_single_proofs {
            vec![Vec::with_capacity(self.depth); indices.len()]
        } else {
            Vec::new()
        };

        // Repeatedly compute the next layer hashes and indices.
        let mut depth = self.depth;
        let mut layer = Vec::with_capacity(2 * leaves.len());
        let mut next_indices = Vec::with_capacity(indices.len());
        let mut next_layer = Vec::new();

        while depth > 0 {
            // Complete the current layer, adding missing siblings
            assert_eq!(indices.len(), leaves.len());
            let mut indices_it = indices.iter().copied().peekable();
            let mut leaves_it = leaves.iter().copied();
            while let Some(index) = indices_it.next() {
                let leaf = leaves_it.next().unwrap();
                if index % 2 == 0 {
                    // Left child, check if next leaf is right child else fetch a sibling.
                    layer.push(leaf);
                    layer.push(if indices_it.peek().copied() == Some(index + 1) {
                        // It is, merge two siblings
                        indices_it.next();
                        leaves_it.next().unwrap()
                    } else {
                        // It is not, fetch a sibling.
                        siblings.next().ok_or(Error::InsufficientSiblings)?
                    });
                } else {
                    // Right child (and we didn't merge in last round), fetch left sibling
                    layer.push(siblings.next().ok_or(Error::InsufficientSiblings)?);
                    layer.push(leaf);
                }
                next_indices.push(index / 2);
            }
            // Store siblings for single proofs
            if produce_single_proofs {
                for (&index, proof) in zip(leaf_indices.iter(), single_proofs.iter_mut()) {
                    let index = index >> (self.depth - depth);
                    let position = next_indices.iter().position(|&i| i == index >> 1).unwrap();
                    proof.push(layer[position * 2 + (index & 1)]);
                }
            }

            // Compute the next layer hashes
            assert_eq!(layer.len() % 2, 0);
            next_layer.resize(layer.len() / 2, HASH_ZERO);
            
            self.hasher_at_depth(depth)
                .hash_pairs(&layer, &mut next_layer);
            
            // Repeat loop, re-using vecs
            depth -= 1;
            swap(&mut indices, &mut next_indices);
            swap(&mut leaves, &mut next_layer);
            next_indices.clear();
            next_layer.clear();
            layer.clear();
        }

        // Make sure we consumed all siblings
        if siblings.next().is_some() {
            return Err(Error::ExcessSiblings);
        }

        // Check root
        assert!(leaves.len() == 1);
        if leaves[0] != *root {
            return Err(Error::RootMismatch);
        }

        Ok(single_proofs)
    }

    /// Sort and deduplicate indices and leaves.
    fn sort_indices(
        &self,
        indices: &[usize],
        leaves: &[Hash],
    ) -> Result<(Vec<usize>, Vec<Hash>), Error> {
        // Ensure indices are in-bounds
        let index_bound = self.size_at_depth(self.depth);
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
            //     if leaves.last() != Some(&leaf) {
            //         return Err(Error::LeafMismatch);
            //     }

            } else {
                indices.push(i);
                leaves.push(leaf);
            }
        }

        Ok((indices, leaves))
    }
}
