use ark_ff::Field;
use spongefish::codecs::arkworks_algebra::FieldDomainSeparator;

use crate::fs_utils::WhirPoWDomainSeparator;

pub trait SumcheckSingleDomainSeparator<F: Field> {
    #[must_use]
    fn add_sumcheck(self, folding_factor: usize, pow_bits: f64) -> Self;
}

impl<F, DomainSeparator> SumcheckSingleDomainSeparator<F> for DomainSeparator
where
    F: Field,
    DomainSeparator: FieldDomainSeparator<F> + WhirPoWDomainSeparator,
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
