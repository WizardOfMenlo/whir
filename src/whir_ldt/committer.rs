use super::parameters::WhirConfig;
use crate::{
    ntt::expand_from_coeff,
    poly_utils::{coeffs::CoefficientList, fold::restructure_evaluations},
    utils,
};
use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use nimue::{plugins::ark::FieldChallenges, ByteWriter, Merlin, ProofResult};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    pub(crate) merkle_leaves: Vec<F>,
}

pub struct Committer<F, MerkleConfig, PowStrategy>(WhirConfig<F, MerkleConfig, PowStrategy>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig, PowStrategy> Committer<F, MerkleConfig, PowStrategy>
where
    F: FftField,
    MerkleConfig: Config<Leaf = [F]>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
{
    pub fn new(config: WhirConfig<F, MerkleConfig, PowStrategy>) -> Self {
        Self(config)
    }

    pub fn commit(
        &self,
        merlin: &mut Merlin,
        polynomial: CoefficientList<F::BasePrimeField>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        let base_domain = self.0.starting_domain.base_domain.unwrap();
        let expansion = base_domain.size() / polynomial.num_coeffs();
        let evals = expand_from_coeff(polynomial.coeffs(), expansion);
        // TODO: `stack_evaluations` and `restructure_evaluations` are really in-place algorithms.
        // They also partially overlap and undo one another. We should merge them.
        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);
        let folded_evals = restructure_evaluations(
            folded_evals,
            self.0.fold_optimisation,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor,
        );

        // Convert to extension field.
        // This is not necessary for the commit, but in further rounds
        // we will need the extension field. For symplicity we do it here too.
        // TODO: Commit to base field directly.
        let folded_evals = folded_evals
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

        merlin.add_bytes(root.as_ref())?;

        let polynomial = polynomial.to_extension();
        Ok(Witness {
            polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
        })
    }
}
