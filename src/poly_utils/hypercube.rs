#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BinaryHypercubePoint(pub usize);

pub struct BinaryHypercube {
    pos: usize,
    num_variables: usize,
}

impl BinaryHypercube {
    pub fn new(num_variables: usize) -> Self {
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
