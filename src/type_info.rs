//! Types that can provide serializable type information for identification.

use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
};

use ark_ff::{Field, PrimeField};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
use zerocopy::IntoBytes;

/// Types that can provide serializable type information for identification.
pub trait TypeInfo {
    type Info: Debug + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

    fn type_info() -> Self::Info;
}

/// Type information for a finite field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field characteristic (aka prime or modulus) in big-endian without leading zeros.
    #[serde(with = "crate::ark_serde::bytes")]
    characteristic: Vec<u8>,

    /// Extension degree of the field.
    degree: usize,
}

/// Zero-sized type that serializes into [`TypeInfo::type_info`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Type<T: TypeInfo>(PhantomData<T>);

impl<T: TypeInfo> Type<T> {
    /// Creates a new type instance.
    pub const fn new() -> Self {
        Type(PhantomData)
    }
}

impl<T: TypeInfo> Debug for Type<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        T::type_info().fmt(f)
    }
}

impl<T: TypeInfo> Serialize for Type<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        T::type_info().serialize(serializer)
    }
}

impl<'de, T: TypeInfo> Deserialize<'de> for Type<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let expected = T::type_info();
        let got = T::Info::deserialize(deserializer)?;
        if expected != got {
            Err(D::Error::custom(format!(
                "Type mismatch, expected: {expected:?}, got: {got:?}"
            )))
        } else {
            Ok(Type(PhantomData))
        }
    }
}

impl<F: Field> TypeInfo for F {
    type Info = FieldInfo;

    fn type_info() -> Self::Info {
        let modulus = F::BasePrimeField::MODULUS;
        let limbs = modulus.as_ref(); // Little-endian
        let bytes = limbs.as_bytes();
        let characteristic = bytes
            .iter()
            .copied()
            .rev()
            .skip_while(|&b| b == 0)
            .collect();
        FieldInfo {
            characteristic,
            degree: F::extension_degree() as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::const_assert_eq;

    use super::*;
    use crate::{
        crypto::fields::{Field256, Field64_3},
        utils::test_serde,
    };

    const_assert_eq!(size_of::<Type<Field256>>(), 0);

    #[test]
    fn test_type_info_field64_3() {
        let type_info = Field64_3::type_info();
        assert_eq!(
            type_info.characteristic,
            18446744069414584321_u64.to_be_bytes().as_slice()
        );
        assert_eq!(type_info.degree, 3);
    }

    #[test]
    fn test_json_goldilocks_3() {
        let field_config = Type::<Field64_3>::new();
        let json = serde_json::to_string(&field_config).unwrap();
        assert_eq!(
            json,
            "{\"characteristic\":\"ffffffff00000001\",\"degree\":3}"
        );
    }

    #[test]
    fn test_roundtrip() {
        test_serde(&Type::<Field256>::new());
        test_serde(&Type::<Field64_3>::new());
    }
}
