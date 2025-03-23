use std::ops::{Deref, DerefMut};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
// TODO (Gotti): Should pos rather be a u64? usize is platform-dependent, giving a
// platform-dependent limit on the number of variables. num_variables may be smaller as well.

// NOTE: Conversion BinaryHypercube <-> MultilinearPoint is Big Endian, using only the num_variables
// least significant bits of the number stored inside BinaryHypercube.

/// point on the binary hypercube {0,1}^n for some n.
///
/// The point is encoded via the n least significant bits of a usize in big endian order and we do
/// not store n.
pub struct BinaryHypercubePoint(pub usize);

impl Deref for BinaryHypercubePoint {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BinaryHypercubePoint {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// An iterator over all points of the binary hypercube `{0,1}^n`.
///
/// The hypercube consists of `2^num_variables` points, where `num_variables` represents
/// the number of binary dimensions.
///
/// - Each point is represented as an integer whose binary representation encodes its coordinates in
///   the hypercube.
/// - Iteration produces points in lexicographic order (`0, 1, 2, ...`).
#[derive(Debug)]
pub struct BinaryHypercube {
    /// Current position in the hypercube, encoded using the bits of `pos`.
    pos: usize,
    /// The number of dimensions (`n`) in the hypercube.
    num_variables: usize,
}

impl BinaryHypercube {
    /// Constructs a new iterator for a binary hypercube `{0,1}^num_variables`.
    pub const fn new(num_variables: usize) -> Self {
        // Note that we need strictly smaller, since some code would overflow otherwise.
        debug_assert!(num_variables < usize::BITS as usize);
        Self { pos: 0, num_variables }
    }
}

impl Iterator for BinaryHypercube {
    type Item = BinaryHypercubePoint;

    /// Advances the iterator and returns the next point in the binary hypercube.
    ///
    /// The iteration stops once all `2^num_variables` points have been produced.
    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.pos;
        if curr < (1 << self.num_variables) {
            self.pos += 1;
            Some(BinaryHypercubePoint(curr))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_hypercube_iterator() {
        let mut hypercube = BinaryHypercube::new(2);

        // The hypercube should generate 2^2 = 4 points
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(1)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(2)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(3)));
        assert_eq!(hypercube.next(), None); // End of iteration
    }

    #[test]
    fn test_binary_hypercube_single_dimension() {
        let mut hypercube = BinaryHypercube::new(1);

        // A 1-dimensional hypercube should produce 2 points: 0, 1
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(1)));
        assert_eq!(hypercube.next(), None);
    }

    #[test]
    fn test_binary_hypercube_zero_dimensions() {
        let mut hypercube = BinaryHypercube::new(0);

        // A 0-dimensional hypercube should produce exactly 1 point (the origin)
        assert_eq!(hypercube.next(), Some(BinaryHypercubePoint(0)));
        assert_eq!(hypercube.next(), None);
    }

    #[test]
    fn test_binary_hypercube_large_dimensions() {
        let n = 5;
        let hypercube = BinaryHypercube::new(n);
        let expected_size = 1 << n; // 2^n

        let mut count = 0;
        for _ in hypercube {
            count += 1;
        }

        assert_eq!(count, expected_size);
    }

    #[test]
    fn test_binary_hypercube_point_order() {
        let mut hypercube = BinaryHypercube::new(3);
        let expected_points = vec![
            BinaryHypercubePoint(0),
            BinaryHypercubePoint(1),
            BinaryHypercubePoint(2),
            BinaryHypercubePoint(3),
            BinaryHypercubePoint(4),
            BinaryHypercubePoint(5),
            BinaryHypercubePoint(6),
            BinaryHypercubePoint(7),
        ];

        for expected in expected_points {
            assert_eq!(hypercube.next(), Some(expected));
        }

        // Ensure iteration stops
        assert_eq!(hypercube.next(), None);
    }
}
