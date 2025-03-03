use crate::fs_utils::WhirPoWIOPattern;
use ark_ff::Field;
use nimue::plugins::ark::FieldIOPattern;

pub trait SumcheckSingleIOPattern<F: Field> {
    fn add_sumcheck(self, folding_factor: usize, pow_bits: f64) -> Self;
}

impl<F, IOPattern> SumcheckSingleIOPattern<F> for IOPattern
where
    F: Field,
    IOPattern: FieldIOPattern<F> + WhirPoWIOPattern,
{
    fn add_sumcheck(mut self, folding_factor: usize, pow_bits: f64) -> Self {
        for _ in 0..folding_factor {
            self = self
                .add_scalars(3, "sumcheck_poly")
                .challenge_scalars(1, "folding_randomness")
                .pow(pow_bits);
        }
        self
    }
}
