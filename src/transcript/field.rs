use std::marker::PhantomData;

use ark_ff::{BigInteger, Field, PrimeField};
use serde::ser::{Serialize, SerializeStruct, Serializer};

/// Zero-sized type that represents a Galois field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FieldConfig<F: Field> {
    field: PhantomData<F>,
}

impl<F: Field> FieldConfig<F> {
    pub fn new() -> Self {
        Self { field: PhantomData }
    }
}

impl<F: Field> Serialize for FieldConfig<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let modulus = F::BasePrimeField::MODULUS.to_bytes_be();
        let leading_zeros = modulus.iter().take_while(|&&b| b == 0).count();
        let modulus = &modulus[leading_zeros..];
        let mut s = serializer.serialize_struct("FieldConfig", 2)?;
        s.serialize_field("prime", modulus)?;
        s.serialize_field("extension_degree", &F::extension_degree())?;
        s.end()
    }
}

// TODO: Deserialize and verify field configuration.
