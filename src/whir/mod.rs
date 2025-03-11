use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub mod committer;
pub mod fs_utils;
pub mod iopattern;
pub mod parameters;
pub mod prover;
pub mod statement;
pub mod verifier;


// Only includes the authentication paths
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct WhirProof<MerkleConfig, F>
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    pub merkle_paths: Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>,
    pub statement_values_at_random_point: Vec<F>,
}

pub fn whir_proof_size<MerkleConfig, F>(
    transcript: &[u8],
    whir_proof: &WhirProof<MerkleConfig, F>,
) -> usize
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize,
{
    transcript.len() + whir_proof.serialized_size(ark_serialize::Compress::Yes)
}

#[cfg(test)]
mod tests {
    use nimue::{DefaultHash, IOPattern};
    use nimue_pow::blake3::Blake3PoW;

    use crate::crypto::fields::Field64;
    use crate::crypto::merkle_tree::blake3 as merkle_tree;
    use crate::parameters::{
        FoldType, FoldingFactor, MultivariateParameters, SoundnessType, WhirParameters,
    };
    use crate::poly_utils::coeffs::CoefficientList;
    use crate::poly_utils::evals::EvaluationsList;
    use crate::poly_utils::MultilinearPoint;
    use crate::whir::statement::{Statement, StatementVerifier, Weights};
    use crate::whir::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier,
    };

    type MerkleConfig = merkle_tree::MerkleTreeParams<F>;
    type PowStrategy = Blake3PoW;
    type F = Field64;

    fn make_whir_things(
        num_variables: usize,
        folding_factor: FoldingFactor,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) {
        let num_coeffs = 1 << num_variables;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = merkle_tree::default_config::<F>(&mut rng);

        let mv_params = MultivariateParameters::<F>::new(num_variables);

        let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
            initial_statement: true,
            security_level: 32,
            pow_bits,
            folding_factor,
            leaf_hash_params,
            two_to_one_params,
            soundness_type,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: 1,
            fold_optimisation: fold_type,
        };

        let params = WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params);

        let polynomial = CoefficientList::new(vec![F::from(1); num_coeffs]);

        let points: Vec<_> = (0..num_points)
            .map(|_| MultilinearPoint::rand(&mut rng, num_variables))
            .collect();

        let mut statement = Statement::<F>::new(num_variables);

        for point in &points {
            let eval = polynomial.evaluate(point);
            let weights = Weights::evaluation(point.clone());
            statement.add_constraint(weights, eval);
        }

        let input = CoefficientList::new(
            (0..1<<num_variables)
                .map(F::from)
                .collect(),
        );
        let input : EvaluationsList<F> = input.clone().into();
 
        let linear_claim_weight = Weights::linear(input.clone());
        let poly = EvaluationsList::from(polynomial.clone().to_extension());
        
        let sum = linear_claim_weight.weighted_sum(&poly);
        statement.add_constraint(linear_claim_weight, sum);

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial).unwrap();

        let prover = Prover(params.clone());
        let statement_verifier = StatementVerifier::from_statement(&statement);

        let proof = prover
            .prove(&mut merlin, statement, witness)
            .unwrap();

        let verifier = Verifier::new(params);
        let mut arthur = io.to_arthur(merlin.transcript());
        assert!(verifier.verify(&mut arthur, &statement_verifier, &proof).is_ok());
    }

    #[test]
    fn test_whir() {
        let folding_factors = [1, 2, 3, 4];
        let soundness_type = [
            SoundnessType::ConjectureList,
            SoundnessType::ProvableList,
            SoundnessType::UniqueDecoding,
        ];
        let fold_types = [FoldType::Naive, FoldType::ProverHelps];
        let num_points = [0, 1, 2];
        let pow_bits = [0, 5, 10];

        for folding_factor in folding_factors {
            let num_variables = folding_factor..=3 * folding_factor;
            for num_variable in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_whir_things(
                                    num_variable,
                                    FoldingFactor::Constant(folding_factor),
                                    num_points,
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
}
