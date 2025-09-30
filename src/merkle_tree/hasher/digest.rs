use {
    super::{Hash, Hasher},
    digest::Digest,
    std::marker::PhantomData,
};

/// Wrapper around a `Digest` to implement the `Hasher` trait.
///
/// This allows using any hash in the Rust-Crypto ecosystem,
/// though performance may vary.
pub struct DigestHasher<T: Digest + Send + Sync> {
    _phantom: PhantomData<T>,
}

impl<T: Digest + Send + Sync> DigestHasher<T> {
    pub fn new() -> Self {
        assert_eq!(<T as Digest>::output_size(), size_of::<Hash>());
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Digest + Send + Sync> Hasher for DigestHasher<T> {
    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert_eq!(
            input.len() % size,
            0,
            "Input length not a multiple of message size."
        );
        assert_eq!(input.len(), size * output.len(), "Output length mismatch.");

        for (input, output) in input.chunks_exact(size).zip(output.iter_mut()) {
            let hash = T::digest(input);
            output.copy_from_slice(hash.as_ref());
        }
    }
}
