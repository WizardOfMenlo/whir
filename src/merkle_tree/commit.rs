use crate::utils::workload_size;
use super::*;

#[cfg(feature = "parallel")]
use rayon::join;

impl MerkleTreeHasher {
    pub fn commit(&mut self, leaves: &[Hash]) -> Vec<Hash> {
        assert_eq!(
            leaves.len(),
            self.size_at_depth(self.depth),
            "Incorrect number of leaves"
        );

        // Allocate space for all the nodes
        // TODO: MaybeUninit?
        let mut nodes = vec![HASH_ZERO; leaves.len() - 1];

        // Create a list of layer slices
        let mut layers = Vec::with_capacity(self.depth);

        let mut tail = nodes.as_mut_slice();
        for i in 0..self.depth {
            let (layer, new_tail) = tail.split_at_mut(self.size_at_depth(i));
            layers.push(layer);
            tail = new_tail;
        }

        // Recursively compute the nodes
        // Skip first few layers to get more simd parallelism in computing roots.
        if 2 * leaves.len() <= workload_size::<Hash>() {
            // Small tree
            self.commit_inner(0, leaves, &mut layers);
        } else {
            // Large tree, skip the first few layers (so we have minimum 64 nodes per layer)
            let first_layer_skip = 6; // 64 nodes
            assert!(layers.len() > first_layer_skip, "Depth too small");
            self.commit_inner(first_layer_skip, leaves, &mut layers[first_layer_skip..]);

            // Compute the first few layers
            let (leaves, nodes) = layers[..=first_layer_skip].split_last_mut().unwrap();
            self.commit_inner(0, leaves, nodes);
        }
        self.layers = layers.into_iter().map(|layer| layer.to_vec()).collect();
        nodes
    }

    fn commit_inner(&self, mut depth: usize, leaves: &[Hash], mut nodes: &mut [&mut [Hash]]) {
        if 2 * leaves.len() <= workload_size::<Hash>() {
            // Base case
            depth += nodes.len();
            self.hasher_at_depth(depth)
                .hash_pairs(leaves, nodes.last_mut().unwrap());
            while let Some((leaves, tail)) = nodes.split_last_mut() {
                nodes = tail;
                depth -= 1;
                if let Some(layer) = nodes.last_mut() {
                    self.hasher_at_depth(depth).hash_pairs(leaves, layer);
                }
            }
        } else {
            {
                // Split into roots and left, right subtrees
                let mut nodes_left = Vec::with_capacity(nodes.len());
                let mut nodes_right = Vec::with_capacity(nodes.len());
                for layer in &mut nodes[1..] {
                    let (l, r) = layer.split_at_mut(layer.len() / 2);
                    nodes_left.push(l);
                    nodes_right.push(r);
                }
                let (leaves_left, leaves_right) = leaves.split_at(leaves.len() / 2);

                // Recurse into the subtrees
                #[cfg(not(feature = "parallel"))]
                {
                    self.commit_inner(depth + 1, leaves_left, &mut nodes_left);
                    self.commit_inner(depth + 1, leaves_right, &mut nodes_right);
                }
                #[cfg(feature = "parallel")]
                join(
                    || self.commit_inner(depth + 1, leaves_left, &mut nodes_left),
                    || self.commit_inner(depth + 1, leaves_right, &mut nodes_right),
                );
            }

            // Compute the roots
            let (roots, nodes) = nodes.split_first_mut().unwrap();
            let (leaves, _) = nodes.split_first_mut().unwrap();
            self.hasher_at_depth(depth).hash_pairs(leaves, roots);
        }
        
    }
}
