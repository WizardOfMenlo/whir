use std::{fmt, marker::PhantomData};

use ark_ff::{BigInteger, Field, PrimeField};
use serde::{
    de::{Deserializer, Error},
    ser::Serializer,
    Deserialize, Serialize,
};

/// Zero-sized type that represents a Galois field.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FieldConfig<F: Field> {
    field: PhantomData<F>,
}

impl<F: Field> FieldConfig<F> {
    pub const fn new() -> Self {
        Self { field: PhantomData }
    }
}

impl<F: Field> FieldConfig<F> {
    pub const fn base_field(&self) -> FieldConfig<F::BasePrimeField> {
        FieldConfig::new()
    }

    /// Modulus in little endian without trailing zeros
    // If MODULUS where `static` instead of `const` we could have return a reference.
    pub fn modulus(&self) -> Vec<u8> {
        let mut bytes = F::BasePrimeField::MODULUS.to_bytes_le();
        let trailing = bytes.iter().rev().take_while(|&b| *b == 0).count();
        bytes.truncate(bytes.len() - trailing);
        bytes
    }

    pub fn extension_degree(&self) -> usize {
        F::extension_degree() as usize
    }

    pub fn size_bytes(&self) -> usize {
        let num_modulus_bytes = ((F::BasePrimeField::MODULUS_BIT_SIZE + 7) / 8) as usize;
        num_modulus_bytes * self.extension_degree()
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

impl<F: Field> fmt::Debug for FieldConfig<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FieldConfig")
            .field("prime", &F::BasePrimeField::MODULUS)
            .field("extension_degree", &self.extension_degree())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use spongefish::NargSerialize;

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

    #[test]
    fn test_size() {
        fn test<F: Field>() {
            let config = FieldConfig::<F>::new();
            assert_eq!(
                config.modulus().len() * config.extension_degree(),
                config.size_bytes()
            );
        }
        test::<Field256>();
        test::<Field64_3>();
    }

    #[test]
    fn test_narg_serialize_length() {
        fn test<F: Field + NargSerialize>() {
            let config = FieldConfig::<F>::new();
            let mut buffer = Vec::new();
            F::ZERO.serialize_into_narg(&mut buffer);
            assert_eq!(buffer.len(), config.size_bytes());
        }
        test::<Field256>();
        test::<Field64_3>();
    }
}
