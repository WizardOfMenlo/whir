use crate::algebra::embedding::Embedding;

/// A Matrix Branching Program
///
/// This generalizes over all known succintly evaluable weights vectors.
pub struct MatrixProgram<M: Embedding> {
    pub embedding: M,
    pub variable_order: Vec<usize>,
    pub dimensions: Vec<usize>,
    pub matrices: Vec<(Vec<M::Target>, Vec<M::Target>)>,
}
