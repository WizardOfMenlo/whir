use {
    super::{Hash, Hasher},
    bytemuck::checked::cast_slice,
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
    fn hash_pairs(&self, blocks: &[Hash], out: &mut [Hash]) {
        assert_eq!(blocks.len(), 2 * out.len());
        for (input, output) in blocks.chunks_exact(2).zip(out.iter_mut()) {
            let hash = T::digest(&cast_slice(input));
            output.copy_from_slice(hash.as_ref());
        }
    }
}
