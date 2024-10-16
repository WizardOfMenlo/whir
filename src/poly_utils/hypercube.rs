#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]

// TODO (Gotti): Should pos rather be a u64? usize is platform-dependent, giving a platform-dependent limit on the number of variables.
// num_variables may be smaller as well.

// NOTE: Conversion BinaryHypercube <-> MultilinearPoint is Big Endian, using only the n least significant bits of the number stored inside BinaryHypercube.

/// point on the binary hypercube {0,1}^n for some n.
/// 
/// The point is encoded via the bits of a usize and we do not store n.
pub struct BinaryHypercubePoint(pub usize);

// BinaryHypercube is an Iterator that is used to range over the points of {0,1}^n, where n == `num_variables`
pub struct BinaryHypercube {
    pos: usize,  // current position, encoded via the bits of pos
    num_variables: usize, // dimension of the hypercube
}

impl BinaryHypercube {
    pub fn new(num_variables: usize) -> Self {
        debug_assert!(num_variables < usize::BITS as usize);
        BinaryHypercube {
            pos: 0,
            num_variables,
        }
    }
}

impl Iterator for BinaryHypercube {
    type Item = BinaryHypercubePoint;

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
