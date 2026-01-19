use std::marker::PhantomData;

use ark_ff::{BigInteger, Field, PrimeField};
use serde::{
    de::{Deserializer, Error},
    ser::Serializer,
    Deserialize, Serialize,
};

/// Zero-sized type that represents a Galois field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FieldConfig<F: Field> {
    field: PhantomData<F>,
}

impl<F: Field> FieldConfig<F> {
    pub const fn new() -> Self {
        Self { field: PhantomData }
    }
}

/// Internal helper
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
#[serde(rename = "FieldConfig")]
struct Parameters<BigInt: BigInteger> {
    #[serde(with = "crate::ark_serde::bigint")]
    prime: BigInt,
    extension_degree: u64,
}

impl<F: Field> Serialize for FieldConfig<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Parameters {
            prime: F::BasePrimeField::MODULUS,
            extension_degree: F::extension_degree(),
        }
        .serialize(serializer)
    }
}

impl<'de, F: Field> Deserialize<'de> for FieldConfig<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let expected = Parameters {
            prime: F::BasePrimeField::MODULUS,
            extension_degree: F::extension_degree(),
        };
        let params: Parameters<<F::BasePrimeField as PrimeField>::BigInt> =
            Parameters::deserialize(deserializer)?;
        if expected != params {
            return Err(D::Error::custom("Mismatch in field"));
        }
        Ok(Self::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crypto::fields::{Field256, Field64_3},
        utils::test_serde,
    };

    #[test]
    fn test_json_goldilocks_3() {
        type F = Field64_3;

        let field_config = FieldConfig::<F>::new();
        let json = serde_json::to_string(&field_config).unwrap();
        assert_eq!(
            json,
            "{\"prime\":\"ffffffff00000001\",\"extension_degree\":3}"
        );
    }

    #[test]
    fn test_roundtrip() {
        test_serde(&FieldConfig::<Field256>::new());
        test_serde(&FieldConfig::<Field64_3>::new());
    }
}
