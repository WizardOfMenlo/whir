use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_poly::{univariate::DensePolynomial, EvaluationDomain};
use nimue::{plugins::ark::FieldChallenges, ByteWriter, Merlin, ProofResult};

use crate::{
    poly_utils::{coeffs::CoefficientList, fold::restructure_evaluations},
    utils,
};

use super::parameters::WhirConfig;

pub struct Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    pub(crate) merkle_leaves: Vec<Vec<F>>,
}

pub struct Committer<F, MerkleConfig>(WhirConfig<F, MerkleConfig>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig> Committer<F, MerkleConfig>
where
    F: FftField,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
{
    pub fn new(config: WhirConfig<F, MerkleConfig>) -> Self {
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
        let univariate: DensePolynomial<_> = polynomial.clone().into();
        let evals = univariate.evaluate_over_domain_by_ref(base_domain).evals;

        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);
        let folded_evals = restructure_evaluations(
            folded_evals,
            self.0.fold_optimisation,
            base_domain.group_gen(),
            base_domain.group_gen_inv(),
            self.0.folding_factor,
        )
        .into_iter()
        .map(|x| x.into_iter().map(F::from_base_prime_field).collect()) // Conver to extension
        .collect();

        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            &folded_evals,
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
