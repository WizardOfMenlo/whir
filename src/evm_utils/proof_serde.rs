use crate::crypto::merkle_tree::keccak::KeccakDigest;
use ark_ff::PrimeField;
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

use crate::evm_utils::proof_converter::OpenZeppelinMultiProof;

struct OpenZeppelinMultiProofVisitor {}
impl<'de> Visitor<'de> for OpenZeppelinMultiProofVisitor {
    type Value = OpenZeppelinMultiProof;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "OpenZeppelinMultiProof")
    }

    fn visit_map<A>(
        self,
        mut map: A,
    ) -> Result<OpenZeppelinMultiProof, <A as MapAccess<'de>>::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut proof: Option<Vec<KeccakDigest>> = None;
        let mut proof_flags: Option<Vec<bool>> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "proof" => {
                    proof = Some(
                        map.next_value::<Vec<String>>()?
                            .iter()
                            .map(|hex_str| keccak_from_string(hex_str))
                            .collect(),
                    );
                }
                "proofFlags" => {
                    proof_flags = Some(map.next_value()?);
                }
                _ => {
                    println!("Unknown key: {}", key);
                }
            }
        }

        Ok(OpenZeppelinMultiProof {
            proof: proof.unwrap(),
            proof_flags: proof_flags.unwrap(),
        })
    }
}

fn keccak_from_string(hex_str: &str) -> KeccakDigest {
    //Remove the "0x" prefix
    let hex_str = hex_str.trim_start_matches("0x");
    let bytes = hex::decode(hex_str).unwrap();
    let mut bytes_array = [0; 32];
    bytes_array.copy_from_slice(&bytes);
    KeccakDigest::from(bytes_array)
}

impl<'de> Deserialize<'de> for OpenZeppelinMultiProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(OpenZeppelinMultiProofVisitor {})
    }
}

impl Serialize for OpenZeppelinMultiProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("OpenZeppelinMultiProof", 3)?;

        state.serialize_field(
            "proof",
            &self
                .proof
                .iter()
                // "0x" is necessary for Foundry to recognize it as bytes32
                .map(|digest| "0x".to_owned() + &hex::encode(digest.as_ref()))
                .collect::<Vec<String>>(),
        )?;
        state.serialize_field("proofFlags", &self.proof_flags)?;

        state.end()
    }
}

pub trait EvmFieldElementSerDe {
    /// Serialize a PrimeField to a hex string in EVM format
    fn serialize(&self) -> String;
}

impl<T: PrimeField> EvmFieldElementSerDe for T {
    fn serialize(&self) -> String {
        let mut byte_buf = vec![];
        self.serialize_compressed(&mut byte_buf).unwrap();
        byte_buf.reverse();
        "0x".to_owned() + &hex::encode(byte_buf)
    }
}
