//! Produce challenge indices from a transcript.

use ark_ff::Field;

use crate::{
    algebra::geometric_sequence,
    transcript::{Decoding, VerifierMessage},
};

pub fn geometric_challenge<T, F>(transcript: &mut T, count: usize) -> Vec<F>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    match count {
        0 => Vec::new(),
        1 => vec![F::ONE],
        _ => {
            // Only source entropy when required
            let x = transcript.verifier_message();
            geometric_sequence(x, count)
        }
    }
}
