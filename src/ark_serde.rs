//! Workaround for `ark_ff` lacking Serde support.
//! See <https://github.com/arkworks-rs/algebra/pull/506>

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{de::Error as _, ser::Error as _, Deserialize as _, Deserializer, Serializer};

pub fn serialize<T, S>(obj: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: CanonicalSerialize,
    S: Serializer,
{
    // Convert to bytes
    let mut buf = Vec::with_capacity(obj.compressed_size());
    obj.serialize_compressed(&mut buf)
        .map_err(|e| S::Error::custom(format!("Failed to serialize: {e}")))?;

    // Write bytes
    if serializer.is_human_readable() {
        // ark_serialize doesn't have human-readable serialization. And Serde
        // doesn't have good defaults for [u8]. So we manually implement hexadecimal
        // serialization.
        let hex = hex::encode(buf);
        serializer.serialize_str(&hex)
    } else {
        serializer.serialize_bytes(&buf)
    }
}

pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: CanonicalDeserialize,
    D: Deserializer<'de>,
{
    // Read bytes
    let bytes = if deserializer.is_human_readable() {
        let hex = String::deserialize(deserializer)?;
        hex::decode(hex).map_err(|e| D::Error::custom(format!("while deserializing bytes: {e}")))?
    } else {
        <Vec<u8>>::deserialize(deserializer)?
    };

    // Convert to object
    let mut reader = &*bytes;
    let obj = T::deserialize_compressed(&mut reader)
        .map_err(|e| D::Error::custom(format!("while deserializing: {e}")))?;
    if !reader.is_empty() {
        return Err(D::Error::custom("while deserializing: trailing bytes"));
    }

    Ok(obj)
}
