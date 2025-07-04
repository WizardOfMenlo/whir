pub trait SumcheckPolynomials {
    /// The field type of the MLEs in the sumcheck protocol.
    // OPT: Is there a usecase for mixed-extension degrees?
    type F: Field;

    /// The number of MLEs in the sumcheck protocol.
    const N: usize;

    /// The degree of the polynomial to be checked in the sumcheck protocol.
    const M: usize;

    /// Given MLE values on a corner of the hypercube, produce accumulatable values
    /// for the construction of the sumcheck polynomial.
    fn map(&self, mles: &[Self::F; Self::N]) -> [Self::F; Self::M];

    /// Given a the sum of p(0) + p(1) and the the sum of values produced by the map,
    /// reconstruct the polynomial p(x) at the next round of the sumcheck protocol.
    fn polynomial(&self, previous_sum: Self::F, sums: &[Self::F]) -> [Self::F; Self::M];
}

#[derive(Debug, Clone)]
pub struct Config<F>
where
    F: Field,
{
    /// Size of the MLEs in the sumcheck protocol.
    size: usize,

    /// The degree of the polynomial to be checked in the sumcheck protocol.
    degree: usize,

    /// Number of folds to apply in the sumcheck protocol.
    num_round: usize,
}

pub trait Prover<F>
where
    F: Field,
{
    fn sumcheck_prove(
        &mut self,
        label: Label,
        config: &Config<F>,
        polynomials: &[F; Config::<F>::N],
    ) -> Result<(), VerifierError>;
}
