//! Degree-1 sumcheck protocol.

mod sumcheck_polynomial;
mod sumcheck_single;

pub use self::{
    sumcheck_polynomial::SumcheckPolynomial,
    sumcheck_single::{Config, RoundConfig, SumcheckSingle},
};
