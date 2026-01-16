use std::marker::PhantomData;

use ark_ff::{BigInteger, Field, PrimeField};
use serde::{
    de::{Deserializer, Error},
    ser::Serializer,
    Deserialize, Serialize,
};

/// Zero-sized type that represents a Galois field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FieldConfig<F: Field> {
    #[serde(
        flatten,
        serialize_with = "serialize",
        deserialize_with = "deserialize"
    )]
    field: PhantomData<F>,
}

impl<F: Field> FieldConfig<F> {
    pub fn new() -> Self {
        Self { field: PhantomData }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct Parameters<'a> {
    prime: &'a [u8],
    extension_degree: u64,
}

fn serialize<S, F>(value: &PhantomData<F>, s: S) -> Result<S::Ok, S::Error>
where
    F: Field,
    S: Serializer,
{
    let modulus = F::BasePrimeField::MODULUS.to_bytes_be();
    let leading_zeros = modulus.iter().take_while(|&&b| b == 0).count();
    Parameters {
        prime: &modulus[leading_zeros..],
        extension_degree: F::extension_degree(),
    }
    .serialize(s)
}

fn deserialize<'de, D, F>(d: D) -> Result<PhantomData<F>, D::Error>
where
    F: Field,
    D: Deserializer<'de>,
{
    let modulus = F::BasePrimeField::MODULUS.to_bytes_be();
    let leading_zeros = modulus.iter().take_while(|&&b| b == 0).count();
    let expected = Parameters {
        prime: &modulus[leading_zeros..],
        extension_degree: F::extension_degree(),
    };
    let params = Parameters::deserialize(d)?;
    if expected != params {
        return Err(D::Error::custom(format!("Mismatch in field")));
    }
    Ok(PhantomData)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::fields::Field64_3;

    #[test]
    fn test_json_goldilocks_3() {
        type F = Field64_3;

        let field_config = FieldConfig::<F>::new();
        let json = serde_json::to_string(&field_config).unwrap();
        assert_eq!(
            json,
            "{\"prime\":[255,255,255,255,0,0,0,1],\"extension_degree\":3}"
        );
    }
}
