//! Interactive (sub)protocols for WHIR.
//!
//! These interact through the [`spongefish`] Fiat–Shamir transformation

pub mod challenge_indices;
pub mod irs_commit;
pub mod matrix_commit;
pub mod merkle_tree;
pub mod proof_of_work;
pub mod sumcheck;
