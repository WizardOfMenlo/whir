use super::*;

impl MerkleTreeHasher {
    /// Produce a Merkle proof for the set of indices.
    pub fn proof<F>(&self, indices: &[usize], mut leaves_lookup: F) -> Vec<Hash>
    where
        F: FnMut(usize, usize) -> Hash,
    {
        if indices.is_empty() {
            return Vec::new();
        }

        // Gather leaf hashes and sort/dedup by indices (validates duplicates)
        let mut leaves = Vec::with_capacity(indices.len());
        for &i in indices {
            leaves.push(leaves_lookup(i, self.depth-1));
        }

        let mut indices = indices.to_vec();

        let mut proof: Vec<Hash> = Vec::with_capacity(indices.len());

        // Walk up the tree, emitting missing siblings in the verifier's consumption order
        let mut depth = self.depth - 1;
        while depth > 0 {
            let mut next_indices = Vec::with_capacity(indices.len());
            let mut next_leaves = Vec::with_capacity(indices.len());

            let mut i = 0;
            while i < indices.len() {
                let idx = indices[i];

                if idx % 2 == 0 {
                    // Left child
                    if i + 1 < indices.len() && indices[i + 1] == idx + 1 {
                        // Adjacent right child present -> merge without emitting a sibling
                        let pair = [leaves[i], leaves[i + 1]];
                        let mut out = [HASH_ZERO; 1];
                        self.hasher_at_depth(depth).hash_pairs(&pair, &mut out);
                        next_indices.push(idx / 2);
                        next_leaves.push(out[0]);
                        i += 2;
                    } else {
                        // Missing right sibling -> add it to the proof
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
                    // Right child, missing left sibling -> add it to the proof
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
            leaves = next_leaves;
            depth -= 1;
        }

        proof
    }
}