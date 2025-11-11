use crate::merkle_tree::HASH_ZERO;
use crate::merkle_tree::Hash;

use super::*;

impl MerkleTreeHasher {
    /// Produce a Merkle proof (multi- or single) for the given set of indices.
    pub fn proof<F>(&self, indices: &[usize], mut leaves_lookup: F) -> MerkleProof
    where
        F: FnMut(usize, usize) -> Hash,
    {
        if indices.is_empty() {
            return MerkleProof{
                depth: self.depth, 
                proof: vec![], 
                indices: vec![]
            };
        }

        // Gather leaf hashes and sort/dedup to match verifier’s expectation
        let mut combined = indices
            .iter()
            .copied()
            .map(|i| (i, leaves_lookup(i, self.depth)))
            .collect::<Vec<_>>();
        combined.sort_by_key(|&(i, _)| i);

        combined.dedup_by(|a, b| {
            a.0 == b.0
        });

        let original_indices = indices.to_vec();
        let mut indices = Vec::with_capacity(combined.len());
        let mut leaves  = Vec::with_capacity(combined.len());
        for (i, leaf) in combined {
            indices.push(i);
            leaves.push(leaf);
        }

        let mut proof: Vec<Hash> = Vec::new();

        // Walk up the tree, emitting missing siblings in verifier's order
        let mut depth = self.depth ;
        while depth > 0 {
            let mut next_indices = Vec::with_capacity(indices.len());
            let mut next_leaves  = Vec::with_capacity(indices.len());

            let mut i = 0;
            while i < indices.len() {
                let idx = indices[i];

                if idx % 2 == 0 {
                    // Left child
                    if i + 1 < indices.len() && indices[i + 1] == idx + 1 {
                        // Adjacent right child present -> merge
                        let pair = [leaves[i], leaves[i + 1]];
                        let mut out = [HASH_ZERO; 1];
                        self.hasher_at_depth(depth).hash_pairs(&pair, &mut out);
                        next_indices.push(idx / 2);
                        next_leaves.push(out[0]);
                        i += 2;
                    } else {
                        // Right sibling missing -> push it into proof
                        let sib = leaves_lookup(idx + 1, depth);
                        proof.push(sib);

                        let pair = [leaves[i], sib];
                        let mut out = [HASH_ZERO; 1];
                        self.hasher_at_depth(depth).hash_pairs(&pair, &mut out);
                        next_indices.push(idx / 2);
                        next_leaves.push(out[0]);
                        i += 1;
                    }
                } else {
                    // Right child -> fetch left sibling
                    let sib = leaves_lookup(idx - 1, depth);
                    proof.push(sib);

                    let pair = [sib, leaves[i]];
                    let mut out = [HASH_ZERO; 1];
                    self.hasher_at_depth(depth).hash_pairs(&pair, &mut out);
                    next_indices.push(idx / 2);
                    next_leaves.push(out[0]);
                    i += 1;
                }
            }

            indices = next_indices;
            leaves  = next_leaves;
            depth -= 1;
        }

        MerkleProof{
            depth: self.depth, 
            proof: proof,
            indices: original_indices
        }
    }
}
