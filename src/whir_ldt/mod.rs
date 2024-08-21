use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub mod committer;
pub mod iopattern;
pub mod parameters;
pub mod prover;
pub mod verifier;

// Only includes the authentication paths
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct WhirProof<MerkleConfig>(Vec<(MultiPath<MerkleConfig>, Vec<MerkleConfig::Leaf>)>)
where
    MerkleConfig: Config,
    MerkleConfig::Leaf: Sized + Clone + CanonicalSerialize + CanonicalDeserialize;

pub fn whir_proof_size<MerkleConfig>(
    transcript: &[u8],
    whir_proof: &WhirProof<MerkleConfig>,
) -> usize
where
    MerkleConfig: Config,
    MerkleConfig::Leaf: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    transcript.len() + whir_proof.serialized_size(ark_serialize::Compress::Yes)
}

#[cfg(test)]
mod tests {
    use nimue::{DefaultHash, IOPattern};

    use crate::crypto::fields::Field64;
    use crate::crypto::merkle_tree::blake3 as merkle_tree;
    use crate::parameters::{FoldType, MultivariateParameters, SoundnessType, WhirParameters};
    use crate::poly_utils::coeffs::CoefficientList;
    use crate::whir_ldt::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier,
    };

    type MerkleConfig = merkle_tree::MerkleTreeParams<F>;

    type F = Field64;

    fn make_whir_things(
        num_variables: usize,
        folding_factor: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) {
        let num_coeffs = 1 << num_variables;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = merkle_tree::default_config::<F>(&mut rng);

        let mv_params = MultivariateParameters::<F>::new(num_variables);

        let whir_params = WhirParameters::<MerkleConfig> {
            security_level: 32,
            pow_bits,
            folding_factor,
            leaf_hash_params,
            two_to_one_params,
            fold_optimisation: fold_type,
            soundness_type,
            starting_log_inv_rate: 1,
        };

        let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params);

        let polynomial = CoefficientList::new(vec![F::from(1); num_coeffs]);

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial).unwrap();

        let prover = Prover(params.clone());

        let proof = prover.prove(&mut merlin, witness).unwrap();

        let verifier = Verifier::new(params);
        let mut arthur = io.to_arthur(merlin.transcript());
        assert!(verifier.verify(&mut arthur, &proof).is_ok());
    }

    #[test]
    fn test_whir_ldt() {
        let folding_factors = [1, 2, 3, 4];
        let fold_types = [FoldType::Naive, FoldType::ProverHelps];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variables in num_variables {
                for fold_type in fold_types {
                    for soundness_type in soundness_type {
                        for pow_bits in pow_bits {
                            make_whir_things(
                                num_variables,
                                folding_factor,
                                soundness_type,
                                pow_bits,
                                fold_type,
                            );
                        }
                    }
                }
            }
        }
    }
}
