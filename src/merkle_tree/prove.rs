use super::*;

impl MerkleTreeHasher {
    /// Produce a Merkle proof for the set of indices.
    pub fn proof<F>(&self, indices: &[usize], mut leaves_lookup: F) -> Vec<Hash>
    where
        F: FnMut(usize, usize) -> Hash,
    {
        let mut proof: Vec<Hash> = Vec::with_capacity(indices.len());
        let mut leaves = Vec::with_capacity(indices.len());
        for &i in indices {
            leaves.push(leaves_lookup(i, self.depth));
        }
        todo!()
    }
}
