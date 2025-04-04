use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::{
    polynomial::DenseUVPolynomial, univariate::DensePolynomial, EvaluationDomain, Polynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{
    codecs::arkworks_algebra::{FieldToUnitSerialize, UnitToField},
    ProofResult,
};

use super::Witness;
use crate::{
    ntt::expand_from_coeff, parameters::FoldType, poly_utils::fold::transform_evaluations,
    stir_ldt::parameters::StirConfig, whir::utils::DigestToUnitSerialize,
};

pub struct CommitmentWriter<F, MerkleConfig, PowStrategy>(StirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> CommitmentWriter<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
{
    pub const fn new(config: StirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(config)
    }

    pub fn commit<ProverState>(
        &self,
        merlin: &mut ProverState,
        polynomial: DensePolynomial<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        ProverState: FieldToUnitSerialize<F> + UnitToField<F> + DigestToUnitSerialize<MerkleConfig>,
    {
        let num_coeffs = polynomial.degree() + 1;
        let base_domain = self.0.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / num_coeffs;
        let mut evals = expand_from_coeff(&polynomial.coeffs[..], expansion);
        transform_evaluations(
            evals.as_mut_slice(),
            FoldType::Naive, // This will eventually change to `FoldType::ProverHelps`.
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor,
        );

        // Convert to extension field.
        // This is not necessary for the commit, but in further rounds
        // we will need the extension field. For symplicity we do it here too.
        // TODO: Commit to base field directly.
        let folded_evals = evals
            .into_iter()
            .map(F::from_base_prime_field)
            .collect::<Vec<_>>();

        // Group folds together as a leaf.
        let fold_size = 1 << self.0.folding_factor;
        #[cfg(not(feature = "parallel"))]
        let leafs_iter = folded_evals.chunks_exact(fold_size);
        #[cfg(feature = "parallel")]
        let leafs_iter = folded_evals.par_chunks_exact(fold_size);
        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            leafs_iter,
        )
        .unwrap();

        let root = merkle_tree.root();

        merlin.add_digest(root)?;

        let polynomial = DensePolynomial::from_coefficients_vec(
            polynomial
                .coeffs
                .into_iter()
                .map(F::from_base_prime_field)
                .collect(),
        );

        Ok(Witness {
            polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
        })
    }
}
