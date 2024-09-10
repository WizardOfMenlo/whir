//! NTT and related algorithms.

mod matrix;
mod ntt;
mod transpose;
mod utils;
mod wavelet;

use self::matrix::MatrixMut;
use ark_ff::FftField;

pub use self::{
    ntt::{intt, intt_batch, ntt, ntt_batch},
    transpose::transpose,
    wavelet::wavelet_transform,
};

/// RS encode at a rate 1/`expansion`.
pub fn expand_from_coeff<F: FftField>(coeffs: &[F], expansion: usize) -> Vec<F> {
    let engine = ntt::NttEngine::<F>::new_from_cache();
    let expanded_size = coeffs.len() * expansion;
    let root = engine.root(expanded_size);
    let mut result = Vec::with_capacity(expanded_size);
    for i in 0..expansion {
        if i == 0 {
            result.extend_from_slice(&coeffs);
        } else {
            let root = root.pow([i as u64]);
            let mut offset = F::ONE;
            result.extend(coeffs.iter().map(|x| {
                let val = *x * offset;
                offset *= root;
                val
            }));
        }
    }
    ntt_batch(&mut result, coeffs.len());
    transpose(&mut result, expansion, coeffs.len());
    result
}
