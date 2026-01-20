use std::marker::PhantomData;

use digest::{const_oid::AssociatedOid, Digest};
use hex_literal::hex;
use sha3::Sha3_256;
use zerocopy::FromBytes;

use super::{threshold, Engine};
use crate::{
    protocols::proof_of_work::find_min,
    transcript::{Protocol, ProtocolId},
};

pub const SHA2: ProtocolId = ProtocolId::new(hex!(
    "464fbbfe08764efed04846fa28c6224f99f02a2c8fb1015973e0c9d8957d4c09"
));
pub const SHA3: ProtocolId = ProtocolId::new(hex!(
    "5c9a38cc03e01ae57a8ea270e68acdbe4b97540944fe51789717a4b7a2323ccc"
));

pub type Sha2 = DigestEngine<sha2::Sha256>;
pub type Sha3 = DigestEngine<sha3::Sha3_256>;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DigestEngine<D>
where
    D: AssociatedOid + Digest + Sync + Send,
{
    _digest: PhantomData<D>,
}

impl<D> DigestEngine<D>
where
    D: AssociatedOid + Digest + Sync + Send,
{
    pub fn new() -> Self {
        assert!(<D as Digest>::output_size() > 8);
        Self {
            _digest: PhantomData,
        }
    }
}

impl<D> Engine for DigestEngine<D>
where
    D: AssociatedOid + Digest + Sync + Send,
{
    fn check(&self, challenge: [u8; 32], difficulty: f64, nonce: u64) -> bool {
        let mut hasher = D::new();
        hasher.update(challenge);
        hasher.update(nonce.to_le_bytes());
        let hash = hasher.finalize();
        let (prefix, _) = u64::read_from_prefix(&hash).expect("Hash must be at least 64 bit.");
        prefix < threshold(difficulty)
    }

    fn solve(&self, challenge: [u8; 32], difficulty: f64) -> Option<u64> {
        find_min(|nonce| self.check(challenge, difficulty, nonce))
    }
}

impl<D> Protocol for DigestEngine<D>
where
    D: AssociatedOid + Digest + Sync + Send,
{
    fn protocol_id(&self) -> crate::transcript::ProtocolId {
        let mut hasher = Sha3_256::new();
        hasher.update(b"whir::protocols::proof_of_work::DigestEngine");
        hasher.update(D::OID.as_bytes());
        let hash: [u8; 32] = hasher.finalize().into();
        hash.into()
    }
}

#[cfg(test)]
mod tests {
    use spongefish::{domain_separator, session};

    use super::*;
    use crate::{bits::Bits, protocols::proof_of_work::Config};

    #[test]
    fn protocol_id() {
        assert_eq!(Sha2::new().protocol_id(), SHA2);
        assert_eq!(Sha3::new().protocol_id(), SHA3);
    }

    #[test]
    fn test_sha2() {
        let config = Config {
            engine_id: SHA2,
            difficulty: Bits::new(4.0),
        };
        let ds = domain_separator!("whir::protocols::proof_of_work")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&0_u32);

        let mut prover_state = ds.std_prover();
        config.prove(&mut prover_state);
        let proof = prover_state.narg_string();

        let mut verifier_state = ds.std_verifier(proof);
        config.verify(&mut verifier_state).unwrap();
    }

    #[test]
    fn test_sha3() {
        let config = Config {
            engine_id: SHA3,
            difficulty: Bits::new(4.0),
        };
        let ds = domain_separator!("whir::protocols::proof_of_work")
            .session(session!("Test at {}:{}", file!(), line!()))
            .instance(&0_u32);

        let mut prover_state = ds.std_prover();
        config.prove(&mut prover_state);
        let proof = prover_state.narg_string();

        let mut verifier_state = ds.std_verifier(proof);
        config.verify(&mut verifier_state).unwrap();
    }
}
