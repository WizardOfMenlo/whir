//! Generate indices in a range [0.. limit)

use spongefish::{
    codecs::{BytesCommon, BytesPattern},
    transcript::{Label, Length},
    Unit, UnitCommon, UnitPattern,
};

pub trait ChallengeIndicesPattern<U>: UnitPattern<U>
where
    U: Unit,
{
    fn challenge_indices(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<(), Self::Error>;
}

pub trait ChallengeIndicesCommon<U>: UnitCommon<U>
where
    U: Unit,
{
    fn challenge_indices_out(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        out: &mut [usize],
    ) -> Result<(), Self::Error>;

    fn challenge_indices_array<const N: usize>(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
    ) -> Result<[usize; N], Self::Error> {
        let mut out = [0; N];
        self.challenge_indices_out(label, limit, &mut out)?;
        Ok(out)
    }

    fn challenge_indices_vec(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<Vec<usize>, Self::Error> {
        let mut out = vec![0; size];
        self.challenge_indices_out(label, limit, &mut out)?;
        Ok(out)
    }
}

impl<U, P> ChallengeIndicesPattern<U> for P
where
    U: Unit,
    P: BytesPattern<U>,
{
    fn challenge_indices(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<(), Self::Error> {
        assert!(limit.is_power_of_two(), "Limit must be a power of two");
        let label = label.into();
        self.begin_challenge::<[usize]>(label.clone(), Length::Fixed(size))?;
        let bytes_per_index = limit.trailing_zeros().div_ceil(8) as usize;
        self.challenge_bytes("indices-bytes", size * bytes_per_index)?;
        self.end_challenge::<[usize]>(label, Length::Fixed(size))
    }
}

impl<U, P> ChallengeIndicesCommon<U> for P
where
    U: Unit,
    P: BytesCommon<U>,
{
    fn challenge_indices_out(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        out: &mut [usize],
    ) -> Result<(), Self::Error> {
        assert!(limit.is_power_of_two(), "Limit must be a power of two");
        let label = label.into();
        self.begin_challenge::<[usize]>(label.clone(), Length::Fixed(out.len()))?;
        assert!(limit.is_power_of_two(), "Limit must be a power of two");
        let bytes_per_index = limit.trailing_zeros().div_ceil(8) as usize;
        let bytes = self.challenge_bytes_vec("indices-bytes", out.len() * bytes_per_index)?;
        for (out, chunk) in out.iter_mut().zip(bytes.chunks_exact(bytes_per_index)) {
            *out = chunk.iter().fold(0usize, |acc, &b| (acc << 8) | b as usize) % limit;
        }
        self.end_challenge::<[usize]>(label, Length::Fixed(out.len()))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use spongefish::{transcript::TranscriptRecorder, ProverState, VerifierState};

    use super::*;

    #[test]
    fn test_all_ops() -> Result<()> {
        let mut pattern: TranscriptRecorder = TranscriptRecorder::new();
        pattern.challenge_indices("1", 256, 5)?;
        let pattern = pattern.finalize()?;
        eprintln!("{pattern}");

        let mut prover: ProverState = ProverState::from(&pattern);
        assert_eq!(
            prover.challenge_indices_array::<5>("1", 256)?,
            [143, 160, 54, 111, 165]
        );
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        assert_eq!(
            verifier.challenge_indices_array::<5>("1", 256)?,
            [143, 160, 54, 111, 165]
        );
        verifier.finalize()?;

        Ok(())
    }
}
