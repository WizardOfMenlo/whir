use std::convert::From;
use std::fmt::Debug;

#[derive(Debug)]
pub struct ProofError(nimue::ProofError);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    ProofError(#[from] ProofError),
}

impl From<nimue::ProofError> for ProofError {
    fn from(value: nimue::ProofError) -> Self {
        Self(value)
    }
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ProofError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
