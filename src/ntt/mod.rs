//! NTT and related algorithms.

mod matrix;
mod ntt;
mod transpose;
mod utils;
mod wavelet;

use self::matrix::MatrixMut;

pub use self::{
    ntt::{intt, intt_batch, ntt, ntt_batch},
    transpose::transpose,
    wavelet::wavelet_transform,
};
