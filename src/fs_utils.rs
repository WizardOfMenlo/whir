use ark_ff::Field;
use nimue::{
    plugins::{ark::FieldIOPattern, pow::PoWIOPattern},
    IOPattern,
};

pub trait OODIOPattern<F: Field> {
    fn add_ood(self, num_samples: usize) -> Self;
}

impl<F> OODIOPattern<F> for IOPattern
where
    F: Field,
    IOPattern: FieldIOPattern<F>,
{
    fn add_ood(self, num_samples: usize) -> Self {
        if num_samples > 0 {
            self.challenge_scalars(num_samples, "ood_query")
                .add_scalars(num_samples, "ood_ans")
        } else {
            self
        }
    }
}

pub trait WhirPoWIOPattern {
    fn pow(self, bits: f64) -> Self;
}

impl WhirPoWIOPattern for IOPattern
where
    IOPattern: PoWIOPattern,
{
    fn pow(self, bits: f64) -> Self {
        if bits > 0. {
            self.challenge_pow("pow_queries")
        } else {
            self
        }
    }
}
