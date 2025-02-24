use crate::utils::dedup;
use ark_crypto_primitives::merkle_tree::Config;
use nimue::{ByteChallenges, ProofResult};

pub fn get_challenge_stir_queries<T>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    transcript: &mut T,
) -> ProofResult<Vec<usize>>
where
    T: ByteChallenges,
{
    let folded_domain_size = domain_size / (1 << folding_factor);
    // How many bytes do we need to represent an index in the folded domain?
    // domain_size_bytes = log2(folded_domain_size) / 8
    // (both operations are rounded up)
    let domain_size_bytes = ((folded_domain_size * 2 - 1).ilog2() as usize + 7) / 8;
    // We need these many bytes to represent the query indices
    let mut queries = vec![0u8; num_queries * domain_size_bytes];
    transcript.fill_challenge_bytes(&mut queries)?;
    let indices = queries.chunks_exact(domain_size_bytes).map(|chunk| {
        let mut result = 0;
        for byte in chunk {
            result <<= 8;
            result |= *byte as usize;
        }
        result % folded_domain_size
    });
    Ok(dedup(indices))
}

pub trait DigestWriter<MerkleConfig: Config> {
    fn add_digest(&mut self, digest: MerkleConfig::InnerDigest) -> ProofResult<()>;
}

pub trait DigestReader<MerkleConfig: Config> {
    fn read_digest(&mut self) -> ProofResult<MerkleConfig::InnerDigest>;
}
