use {
    super::{Hash, Hasher},
    crate::utils::as_chunks_exact,
    arrayvec::ArrayVec,
    blake3::{
        guts::{BLOCK_LEN, CHUNK_LEN},
        platform::{Platform, MAX_SIMD_DEGREE},
        IncrementCounter, OUT_LEN,
    },
    bytemuck::cast_slice_mut,
    std::{iter::zip, mem::size_of},
};

// Static assertions
const _: () = assert!(
    OUT_LEN == size_of::<Hash>(),
    "Blake3 compression output does not equal hash size."
);
const _: () = assert!(
    BLOCK_LEN == 2 * size_of::<Hash>(),
    "Blake3 compression input does not equal a pair of hashes."
);
const _: () = assert!(
    CHUNK_LEN == 16 * BLOCK_LEN,
    "Blake3 chunk len is not 16 blocks."
);

/// Default Blake3 initialization vector. Copied here because it is not publicly exported.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Flags for a single block message. Copied here because it is not publicly exported.
const FLAGS_START: u8 = 1 << 0; // CHUNK_START
const FLAGS_END: u8 = 1 << 1; // CHUNK_END
const FLAGS: u8 = 1 << 3; // ROOT

pub struct Blake3Hasher {
    platform: Platform,
}

impl Blake3Hasher {
    pub fn new() -> Self {
        Self {
            platform: Platform::detect(),
        }
    }

    fn hash_many_const<const N: usize>(&self, inputs: &[u8], output: &mut [u8]) {
        // Cast the input to a slice of N-sized arrays.
        let inputs = as_chunks_exact::<u8, N>(inputs);

        // Process up to MAX_SIMD_DEGREE messages in parallel.
        for (inputs, out) in zip(
            inputs.chunks(MAX_SIMD_DEGREE),
            output.chunks_mut(OUT_LEN * MAX_SIMD_DEGREE),
        ) {
            // Construct an array of references to input messages.
            let inputs = inputs
                .iter()
                .collect::<ArrayVec<&[u8; N], MAX_SIMD_DEGREE>>();

            // Hash the messages in parallel.
            self.platform.hash_many::<N>(
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
    }
}

impl Hasher for Blake3Hasher {
    /// Hash many short block-alligned messages in parallel.
    /// Messages must be padded to full block lengths and can not exceed one chunk.
    fn hash_many(&self, size: usize, inputs: &[u8], output: &mut [Hash]) {
        assert!(
            size % BLOCK_LEN == 0,
            "Message size must be a multiple of the block length."
        );
        assert!(
            size <= CHUNK_LEN,
            "Message size must not exceed a single chunk."
        );
        assert!(
            inputs.len() % size == 0,
            "Input size must be a multiple of the message size."
        );
        assert_eq!(output.len(), inputs.len() / size, "Output size mismatch.");
        let blocks = size / BLOCK_LEN;
        let output = cast_slice_mut::<Hash, u8>(output);

        // Undo the monomorphization that Blake3 has in their API.
        match blocks {
            0 => {}
            1 => self.hash_many_const::<{ BLOCK_LEN }>(inputs, output),
            2 => self.hash_many_const::<{ 2 * BLOCK_LEN }>(inputs, output),
            3 => self.hash_many_const::<{ 3 * BLOCK_LEN }>(inputs, output),
            4 => self.hash_many_const::<{ 4 * BLOCK_LEN }>(inputs, output),
            5 => self.hash_many_const::<{ 5 * BLOCK_LEN }>(inputs, output),
            6 => self.hash_many_const::<{ 6 * BLOCK_LEN }>(inputs, output),
            7 => self.hash_many_const::<{ 7 * BLOCK_LEN }>(inputs, output),
            8 => self.hash_many_const::<{ 8 * BLOCK_LEN }>(inputs, output),
            9 => self.hash_many_const::<{ 9 * BLOCK_LEN }>(inputs, output),
            10 => self.hash_many_const::<{ 10 * BLOCK_LEN }>(inputs, output),
            11 => self.hash_many_const::<{ 11 * BLOCK_LEN }>(inputs, output),
            12 => self.hash_many_const::<{ 12 * BLOCK_LEN }>(inputs, output),
            13 => self.hash_many_const::<{ 13 * BLOCK_LEN }>(inputs, output),
            14 => self.hash_many_const::<{ 14 * BLOCK_LEN }>(inputs, output),
            15 => self.hash_many_const::<{ 15 * BLOCK_LEN }>(inputs, output),
            16 => self.hash_many_const::<{ 16 * BLOCK_LEN }>(inputs, output),
            _ => unreachable!("Invalid block count."),
        }
    }
}

#[test]
fn test_digest_pairs_equivalent() {
    use {
        super::{test_pairs_equivalent, DigestHasher},
        blake3::Hasher,
    };
    test_pairs_equivalent(&DigestHasher::<Hasher>::new(), &Blake3Hasher::new());
}
