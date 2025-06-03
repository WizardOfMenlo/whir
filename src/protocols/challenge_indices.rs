//! Generate indices in a range [0.. limit)

use spongefish::{
    codecs::bytes,
    transcript::{self, InteractionError, Label, Length, TranscriptError},
};

pub trait Pattern {
    fn challenge_indices(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<(), TranscriptError>;
}

pub trait Common {
    fn challenge_indices_out(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        out: &mut [usize],
    ) -> Result<(), InteractionError>;

    fn challenge_indices_array<const N: usize>(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
    ) -> Result<[usize; N], InteractionError> {
        let mut out = [0; N];
        self.challenge_indices_out(label, limit, &mut out)?;
        Ok(out)
    }

    fn challenge_indices_vec(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<Vec<usize>, InteractionError> {
        let mut out = vec![0; size];
        self.challenge_indices_out(label, limit, &mut out)?;
        Ok(out)
    }
}

pub use Common as Prover;
pub use Common as Verifier;

impl<P> Pattern for P
where
    P: transcript::Pattern + bytes::Pattern,
{
    fn challenge_indices(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        size: usize,
    ) -> Result<(), TranscriptError> {
        assert!(limit.is_power_of_two(), "Limit must be a power of two");
        let label = label.into();
        self.begin_challenge::<[usize]>(label.clone(), Length::Fixed(size))?;
        let bytes_per_index = limit.trailing_zeros().div_ceil(8) as usize;
        self.challenge_bytes("indices-bytes", size * bytes_per_index)?;
        self.end_challenge::<[usize]>(label, Length::Fixed(size))
    }
}

impl<P> Common for P
where
    P: transcript::Common + bytes::Common,
{
    fn challenge_indices_out(
        &mut self,
        label: impl Into<Label>,
        limit: usize,
        out: &mut [usize],
    ) -> Result<(), InteractionError> {
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
            [30, 163, 209, 110, 10]
        );
        let proof = prover.finalize()?;
        assert_eq!(hex::encode(&proof), "");

        let mut verifier: VerifierState = VerifierState::new(pattern.into(), &proof);
        assert_eq!(
            verifier.challenge_indices_array::<5>("1", 256)?,
            [30, 163, 209, 110, 10]
        );
        verifier.finalize()?;

        Ok(())
    }
}
