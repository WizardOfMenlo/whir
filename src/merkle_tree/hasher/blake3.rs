use {
    super::{Hash, Hasher},
    blake3::{
        guts::BLOCK_LEN,
        platform::{Platform, MAX_SIMD_DEGREE},
        IncrementCounter, OUT_LEN,
    },
    bytemuck::{cast_mut, cast_ref, cast_slice, cast_slice_mut},
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

/// Default Blake3 initialization vector. Copied here because it is not publicly exported.
const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Flags for a single block message. Copied here because it is not publicly exported.
const BLAKE3_FLAGS: u8 = 0x0B; // CHUNK_START | CHUNK_END | ROOT

const SIMD_INPUT_SIZE: usize = MAX_SIMD_DEGREE * BLOCK_LEN;
const SIMD_OUTPUT_SIZE: usize = MAX_SIMD_DEGREE * OUT_LEN;

pub struct Blake3Hasher {
    platform: Platform,
}

impl Blake3Hasher {
    pub fn new() -> Self {
        Self {
            platform: Platform::detect(),
        }
    }

    fn hash_pairs_simd(&self, input: &[u8; SIMD_INPUT_SIZE], output: &mut [u8; SIMD_OUTPUT_SIZE]) {
        // `hash_many` requires an array of references to input messages, instead
        // of a contiguous slice. This is not useful in our case, but there is no
        // alternative API.
        let inputs: [&[u8; BLOCK_LEN]; MAX_SIMD_DEGREE] = std::array::from_fn(|i| {
            input[(i * BLOCK_LEN)..((i + 1) * BLOCK_LEN)]
                .try_into()
                .unwrap()
        });
        self.platform.hash_many::<BLOCK_LEN>(
            &inputs,
            &BLAKE3_IV,
            0,
            IncrementCounter::No,
            BLAKE3_FLAGS,
            0,
            0,
            output,
        );
    }
}

impl Hasher for Blake3Hasher {
    fn hash_pairs(&self, blocks: &[Hash], out: &mut [Hash]) {
        assert_eq!(blocks.len(), 2 * out.len());
        let simd_len = (out.len() / MAX_SIMD_DEGREE) * MAX_SIMD_DEGREE;
        let (input_simd, input) = blocks.split_at(2 * simd_len);
        let (output_simd, output) = out.split_at_mut(simd_len);
        for (input, output) in zip(
            input_simd.chunks_exact(2 * MAX_SIMD_DEGREE),
            output_simd.chunks_exact_mut(MAX_SIMD_DEGREE),
        ) {
            // Full SIMD blocks
            let input: &[Hash; 2 * MAX_SIMD_DEGREE] = input.try_into().unwrap();
            let input: &[u8; SIMD_INPUT_SIZE] = cast_ref(input);
            let output: &mut [Hash; MAX_SIMD_DEGREE] = output.try_into().unwrap();
            let output: &mut [u8; SIMD_OUTPUT_SIZE] = cast_mut(output);
            self.hash_pairs_simd(input, output);
        }
        if !input.is_empty() {
            // Remaining partial block (if any)
            let input: &[u8] = cast_slice(input);
            let output: &mut [u8] = cast_slice_mut(output);
            let mut input_block = [0; SIMD_INPUT_SIZE];
            let mut output_block = [0; SIMD_OUTPUT_SIZE];
            input_block[..input.len()].copy_from_slice(input);
            self.hash_pairs_simd(&input_block, &mut output_block);
            output.copy_from_slice(&output_block[..output.len()]);
        }
    }
}

#[test]
fn test_digest_equivalent() {
    use {
        super::{test_equivalent, DigestHasher},
        blake3::Hasher,
    };
    test_equivalent(&DigestHasher::<Hasher>::new(), &Blake3Hasher::new());
}
