//! Wrapper to expose the `hash_many` function from Blake3.

use std::borrow::Cow;

use arrayvec::ArrayVec;
use blake3::{
    guts::{BLOCK_LEN, CHUNK_LEN},
    platform::{Platform, MAX_SIMD_DEGREE},
    IncrementCounter, OUT_LEN,
};
use const_oid::ObjectIdentifier;
use hex_literal::hex;
use static_assertions::const_assert_eq;
use zerocopy::{transmute_mut, FromBytes};

use super::{Engine, Hash, HASH_COUNTER};
use crate::transcript::ProtocolId;

pub const BLAKE3: ProtocolId = ProtocolId::new(hex!(
    "22d025f72260571abf51e1f9b3ed6f0d27da1c765e1511ae778cea025d6345fd"
));

const EMPTY_HASH: Hash = Hash(hex!(
    "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
));

/// Default Blake3 initialization vector. Copied here because it is not publicly exported.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Flags for a single chunk message. Copied here because it is not publicly exported.
const FLAGS_START: u8 = 1 << 0; // CHUNK_START
const FLAGS_END: u8 = (1 << 1) | (1 << 3); // CHUNK_END | ROOT
const FLAGS: u8 = 0;

// OUT_LEN is the length of the hash output.
const_assert_eq!(OUT_LEN, 32);

// BLOCK_LEN is the input length of the inner compression function.
const_assert_eq!(BLOCK_LEN, 2 * OUT_LEN);

// A `chunk` is the maximum number of blocks in a
const_assert_eq!(CHUNK_LEN, 16 * BLOCK_LEN);

#[derive(Clone, Copy, Debug)]
pub struct Blake3(Platform);

impl Blake3 {
    pub fn new(platform: Platform) -> Self {
        Self(platform)
    }

    pub fn detect() -> Self {
        Self(Platform::detect())
    }

    pub fn supports_size(size: usize) -> bool {
        // Padding is not supported, neither is handling messages larger than a chunk.
        size.is_multiple_of(BLOCK_LEN) && size <= CHUNK_LEN
    }
}

impl Engine for Blake3 {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        "Blake3".into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        // Blake3 has no OID assigned yet as of writing.
        None
    }

    fn supports_size(&self, size: usize) -> bool {
        Self::supports_size(size)
    }

    fn preferred_batch_size(&self) -> usize {
        self.0.simd_degree()
    }

    fn hash_many(&self, size: usize, input: &[u8], out: &mut [super::Hash]) {
        hash_many(&self.0, size, input, out);
    }
}

fn hash_many(platform: &Platform, size: usize, inputs: &[u8], output: &mut [Hash]) {
    assert!(
        size.is_multiple_of(BLOCK_LEN),
        "Message size ({size}) must be a multiple of the block length ({BLOCK_LEN} bytes)."
    );
    assert!(
        size <= CHUNK_LEN,
        "Message size ({size}) must not exceed a single chunk ({CHUNK_LEN} bytes)."
    );
    assert_eq!(
        inputs.len(),
        size * output.len(),
        "Input length should be size * output.len() = {size} * {}",
        output.len()
    );
    if size == 0 {
        output.fill(EMPTY_HASH);
        return;
    }
    let blocks = size / BLOCK_LEN;
    let output = transmute_mut!(output);

    // Undo the monomorphization that Blake3 has in their API.
    match blocks {
        1 => hash_many_const::<{ BLOCK_LEN }>(platform, inputs, output),
        2 => hash_many_const::<{ 2 * BLOCK_LEN }>(platform, inputs, output),
        3 => hash_many_const::<{ 3 * BLOCK_LEN }>(platform, inputs, output),
        4 => hash_many_const::<{ 4 * BLOCK_LEN }>(platform, inputs, output),
        5 => hash_many_const::<{ 5 * BLOCK_LEN }>(platform, inputs, output),
        6 => hash_many_const::<{ 6 * BLOCK_LEN }>(platform, inputs, output),
        7 => hash_many_const::<{ 7 * BLOCK_LEN }>(platform, inputs, output),
        8 => hash_many_const::<{ 8 * BLOCK_LEN }>(platform, inputs, output),
        9 => hash_many_const::<{ 9 * BLOCK_LEN }>(platform, inputs, output),
        10 => hash_many_const::<{ 10 * BLOCK_LEN }>(platform, inputs, output),
        11 => hash_many_const::<{ 11 * BLOCK_LEN }>(platform, inputs, output),
        12 => hash_many_const::<{ 12 * BLOCK_LEN }>(platform, inputs, output),
        13 => hash_many_const::<{ 13 * BLOCK_LEN }>(platform, inputs, output),
        14 => hash_many_const::<{ 14 * BLOCK_LEN }>(platform, inputs, output),
        15 => hash_many_const::<{ 15 * BLOCK_LEN }>(platform, inputs, output),
        16 => hash_many_const::<{ 16 * BLOCK_LEN }>(platform, inputs, output),
        _ => unreachable!("Invalid message size."),
    }
}

fn hash_many_const<const N: usize>(platform: &Platform, inputs: &[u8], output: &mut [u8]) {
    // Cast the input to a slice of N-sized arrays.
    let inputs: &[[u8; N]] =
        <[[u8; N]]>::ref_from_bytes(inputs).expect("Input length is not a multiple of N");

    // Process up to MAX_SIMD_DEGREE messages in parallel.
    for (inputs, out) in inputs
        .chunks(MAX_SIMD_DEGREE)
        .zip(output.chunks_mut(OUT_LEN * MAX_SIMD_DEGREE))
    {
        // Construct an array of references to input messages.
        let inputs = inputs
            .iter()
            .collect::<ArrayVec<&[u8; N], MAX_SIMD_DEGREE>>();

        // Hash the messages in parallel.
        platform.hash_many::<N>(
            &inputs,
            &BLAKE3_IV,
            0,
            IncrementCounter::No,
            FLAGS,
            FLAGS_START,
            FLAGS_END,
            out,
        );
    }
    HASH_COUNTER.add(inputs.len());
}

#[cfg(test)]
mod tests {
    use super::{super::DigestEngine, *};
    use crate::transcript::Protocol;

    #[test]
    fn test_protocol_id() {
        assert_eq!(Blake3::detect().protocol_id(), BLAKE3);
    }

    #[test]
    fn test_eq_digest() {
        let our_engine = Blake3::detect();
        let their_engine = DigestEngine::<blake3::Hasher>::from_name("blake3");

        for n in 0..=16 {
            let size = n * BLOCK_LEN;
            for count in 0..=16 {
                let input: Vec<u8> = (0..size * count).map(|i| i as u8).collect();
                let mut ours = vec![Hash::default(); count];
                let mut theirs = vec![Hash::default(); count];

                our_engine.hash_many(size, &input, &mut ours);
                their_engine.hash_many(size, &input, &mut theirs);

                assert_eq!(ours, theirs);
            }
        }
    }
}
