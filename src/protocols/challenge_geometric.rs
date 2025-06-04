//! Generate field element challenges as a Geometric sequence: 1, r, r^2, r^3, ...

use ark_ff::Field;
use spongefish::{
    codecs::arkworks::field,
    transcript::{self, Label, Length},
};

pub trait Pattern {
    fn challenge_ark_geometric<F>(&mut self, label: impl Into<Label>, size: usize)
    where
        F: Field;
}

pub trait Common {
    fn challenge_ark_geometric_out<F>(&mut self, label: impl Into<Label>, out: &mut [F])
    where
        F: Field;

    fn challenge_ark_geometric_array<F, const N: usize>(
        &mut self,
        label: impl Into<Label>,
    ) -> [F; N]
    where
        F: Field,
    {
        let mut out = [F::default(); N];
        self.challenge_ark_geometric_out(label, &mut out);
        out
    }

    fn challenge_ark_geometric_vec<F>(&mut self, label: impl Into<Label>, size: usize) -> Vec<F>
    where
        F: Field,
    {
        let mut out = vec![F::default(); size];
        self.challenge_ark_geometric_out(label, &mut out);
        out
    }
}

impl<P> Pattern for P
where
    P: transcript::Pattern + field::Pattern,
{
    fn challenge_ark_geometric<F>(&mut self, label: impl Into<Label>, size: usize)
    where
        F: Field,
    {
        let label = label.into();
        self.begin_challenge::<[F]>(label.clone(), Length::Fixed(size));
        if size > 1 {
            self.challenge_ark_fel::<F>("base");
        }
        self.end_challenge::<[F]>(label, Length::Fixed(size))
    }
}

impl<P> Common for P
where
    P: transcript::Common + field::Common,
{
    fn challenge_ark_geometric_out<F>(&mut self, label: impl Into<Label>, out: &mut [F])
    where
        F: Field,
    {
        let label = label.into();
        self.begin_challenge::<[F]>(label.clone(), Length::Fixed(out.len()));
        if !out.is_empty() {
            out[0] = F::ONE;
            if out.len() > 1 {
                let base = self.challenge_ark_fel("base");
                let mut power = base;
                out[1] = base;
                for out in &mut out[2..] {
                    power *= base;
                    *out = power;
                }
            }
        }
        self.end_challenge::<[F]>(label, Length::Fixed(out.len()))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use ark_ff::{Fp, MontBackend, MontConfig};
    use spongefish::{transcript::PatternState, ProverState, VerifierState};

    use super::*;

    /// Configuration for the BabyBear field (modulus = 2^31 - 2^27 + 1, generator = 21).
    #[derive(MontConfig)]
    #[modulus = "2013265921"]
    #[generator = "21"]
    pub struct BabybearMontConfig;
    pub type BabybearConfig = MontBackend<BabybearMontConfig, 1>;
    pub type BabyBear = Fp<BabybearConfig, 1>;

    #[test]
    fn test_all_ops() -> Result<()> {
        let mut pattern: PatternState = PatternState::new();
        pattern.challenge_ark_geometric::<BabyBear>("1", 0);
        pattern.challenge_ark_geometric::<BabyBear>("2", 1);
        pattern.challenge_ark_geometric::<BabyBear>("3", 3);
        let pattern = pattern.finalize();
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        assert_eq!(prover.challenge_ark_geometric_array::<BabyBear, 0>("1"), []);
        assert_eq!(
            prover.challenge_ark_geometric_array::<BabyBear, 1>("2"),
            [BabyBear::ONE]
        );
        assert_eq!(
            prover.challenge_ark_geometric_array::<BabyBear, 3>("3"),
            [
                BabyBear::ONE,
                BabyBear::from(42748529),
                BabyBear::from(1294969904)
            ]
        );
        let proof = prover.finalize();
        assert_eq!(hex::encode(&proof), "");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        assert_eq!(
            verifier.challenge_ark_geometric_array::<BabyBear, 0>("1"),
            []
        );
        assert_eq!(
            verifier.challenge_ark_geometric_array::<BabyBear, 1>("2"),
            [BabyBear::ONE]
        );
        assert_eq!(
            verifier.challenge_ark_geometric_array::<BabyBear, 3>("3"),
            [
                BabyBear::ONE,
                BabyBear::from(42748529),
                BabyBear::from(1294969904)
            ]
        );
        verifier.finalize();

        Ok(())
    }
}
