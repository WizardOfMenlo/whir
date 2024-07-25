use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::{FftField, PrimeField};
use ark_poly::{univariate::DensePolynomial, Polynomial};
use nimue::{
    plugins::ark::{FieldChallenges, FieldWriter},
    ByteWriter, Merlin, ProofResult,
};

use crate::{poly_utils::coeffs::CoefficientList, utils};

use super::parameters::WhirConfig;

pub struct Witness<F, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub(crate) polynomial: CoefficientList<F>,
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    pub(crate) merkle_leaves: Vec<Vec<F>>,
    pub(crate) ood_points: Vec<F>,
}

pub struct Committer<F, MerkleConfig>(WhirConfig<F, MerkleConfig>)
where
    F: FftField,
    MerkleConfig: Config;

impl<F, MerkleConfig> Committer<F, MerkleConfig>
where
    F: FftField + PrimeField,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: AsRef<[u8]>,
{
    pub fn new(config: WhirConfig<F, MerkleConfig>) -> Self {
        Self(config)
    }

    pub fn commit(
        &self,
        merlin: &mut Merlin,
        polynomial: CoefficientList<F>,
    ) -> ProofResult<Witness<F, MerkleConfig>>
    where
        Merlin: FieldChallenges<F> + ByteWriter,
    {
        let univariate: DensePolynomial<_> = polynomial.clone().into();
        let evals = univariate
            .evaluate_over_domain_by_ref(self.0.starting_domain.backing_domain)
            .evals;

        let folded_evals = utils::stack_evaluations(evals, self.0.folding_factor);

        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.0.leaf_hash_params,
            &self.0.two_to_one_params,
            &folded_evals,
        )
        .unwrap();

        let root = merkle_tree.root();

        merlin.add_bytes(root.as_ref())?;

        let mut ood_points = vec![F::ZERO; self.0.committment_ood_samples];
        if self.0.committment_ood_samples > 0 {
            merlin.fill_challenge_scalars(&mut ood_points)?;
            let ood_answers: Vec<_> = ood_points
                .iter()
                .map(|ood_point| univariate.evaluate(&ood_point))
                .collect();
            merlin.add_scalars(&ood_answers)?;
        }

        Ok(Witness {
            polynomial,
            merkle_tree,
            merkle_leaves: folded_evals,
            ood_points,
        })
    }
}
