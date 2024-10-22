use std::marker::PhantomData;

use crate::crypto::{fields::Field256, merkle_tree::keccak::KeccakDigest};
use ark_ff::FftField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize, Serializer,
};

use crate::evm_utils::proof_converter::OpenZeppelinMultiProof;

struct OpenZeppelinMultiProofVisitor<F: FftField> {
    _marker: PhantomData<F>,
}
impl<'de, F: FftField + EvmFieldElementSerDe> Visitor<'de> for OpenZeppelinMultiProofVisitor<F> {
    type Value = OpenZeppelinMultiProof<F>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "OpenZeppelinMultiProof")
    }

    fn visit_map<A>(
        self,
        mut map: A,
    ) -> Result<OpenZeppelinMultiProof<F>, <A as MapAccess<'de>>::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut preimages: Option<Vec<Vec<F>>> = None;
        let mut proof: Option<Vec<KeccakDigest>> = None;
        let mut proof_flags: Option<Vec<bool>> = None;
        let mut root: Option<KeccakDigest> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "preimages" => {
                    preimages = Some(
                        map.next_value::<Vec<Vec<String>>>()?
                            .iter()
                            .map(|inner_vec| {
                                inner_vec.iter().map(|val| F::deserialize(val)).collect()
                            })
                            .collect(),
                    );
                }
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
                "root" => {
                    root = Some(keccak_from_string(&map.next_value::<String>()?));
                }
                _ => {
                    println!("Unknown key: {}", key);
                }
            }
        }

        Ok(OpenZeppelinMultiProof {
            preimages: preimages.unwrap(),
            proof: proof.unwrap(),
            proof_flags: proof_flags.unwrap(),
            root: root.unwrap(),
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

impl<'de, F: FftField + EvmFieldElementSerDe> Deserialize<'de> for OpenZeppelinMultiProof<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(OpenZeppelinMultiProofVisitor::<F> {
            _marker: PhantomData,
        })
    }
}

impl<F: FftField + EvmFieldElementSerDe> Serialize for OpenZeppelinMultiProof<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("OpenZeppelinMultiProof", 3)?;

        state.serialize_field(
            "preimages",
            &self
                .preimages
                .iter()
                .map(|inner_vec| {
                    inner_vec
                        .iter()
                        .map(|val| val.serialize())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<Vec<_>>>(),
        )?;
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
        state.serialize_field(
            "root",
            // "0x" is necessary for Foundry to recognize it as bytes32
            &("0x".to_owned() + &hex::encode(self.root.as_ref())),
        )?;

        state.end()
    }
}

trait EvmFieldElementSerDe {
    fn serialize(&self) -> String;
    fn deserialize(serialized: &str) -> Self;
}

impl EvmFieldElementSerDe for Field256 {
    fn serialize(&self) -> String {
        let mut byte_buf = vec![];
        self.serialize_compressed(&mut byte_buf).unwrap();
        byte_buf.reverse();
        "0x".to_owned() + &hex::encode(byte_buf)
    }

    fn deserialize(serialized: &str) -> Field256 {
        // Remove "0x" prefix
        let val = serialized.trim_start_matches("0x");
        let mut buf = hex::decode(val).unwrap();
        buf.reverse();
        Field256::deserialize_uncompressed(buf.as_slice()).unwrap()
    }
}
