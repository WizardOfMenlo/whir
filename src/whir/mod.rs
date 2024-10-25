use ark_crypto_primitives::merkle_tree::{Config, MultiPath};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{ser::SerializeStruct, Serialize, Serializer};

use crate::evm_utils::proof_serde::EvmFieldElementSerDe;
use crate::poly_utils::MultilinearPoint;

pub mod committer;
pub mod iopattern;
pub mod parameters;
pub mod prover;
pub mod verifier;

#[derive(Debug, Clone)]
pub struct Statement<F> {
    pub points: Vec<MultilinearPoint<F>>,
    pub evaluations: Vec<F>,
}

impl<F: PrimeField> Serialize for MultilinearPoint<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let str_vec: Vec<String> = self.0.iter().map(F::serialize).collect();
        str_vec.serialize(serializer)
    }
}

impl<F: PrimeField> Serialize for Statement<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Statement", 2)?;
        state.serialize_field("points", &self.points)?;
        let evaluations_str: Vec<String> = self.evaluations.iter().map(F::serialize).collect();
        state.serialize_field("evaluations", &evaluations_str)?;
        state.end()
    }
}

// Only includes the authentication paths
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct WhirProof<MerkleConfig, F>(pub(crate) Vec<(MultiPath<MerkleConfig>, Vec<Vec<F>>)>)
where
    MerkleConfig: Config<Leaf = [F]>,
    F: Sized + Clone + CanonicalSerialize + CanonicalDeserialize;

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
mod evm_tests {
    use std::io::Write;

    use crate::crypto::fields::FieldBn256;
    use crate::crypto::merkle_tree::keccak as merkle_tree;
    use crate::evm_utils::hasher::MerkleTreeEvmParams;
    use crate::evm_utils::proof_converter::{convert_whir_proof, FullEvmProof};
    use crate::fs_utils::{EVMFs, KeccakEVMPoW};
    use crate::parameters::{FoldType, MultivariateParameters, SoundnessType, WhirParameters};
    use crate::poly_utils::coeffs::CoefficientList;
    use crate::poly_utils::MultilinearPoint;
    use crate::whir::Statement;
    use crate::whir::{
        committer::Committer, parameters::WhirConfig, prover::Prover, verifier::Verifier,
    };

    use super::WhirProof;

    type MerkleConfig = MerkleTreeEvmParams<F>;
    type PowStrategy = KeccakEVMPoW;
    type F = FieldBn256;

    fn evm_make_whir_things(
        num_variables: usize,
        folding_factor: usize,
        num_points: usize,
        soundness_type: SoundnessType,
        pow_bits: usize,
        fold_type: FoldType,
    ) -> (
        WhirConfig<F, MerkleConfig, PowStrategy>,
        EVMFs<F>,
        Statement<F>,
        WhirProof<MerkleConfig, F>,
    ) {
        let num_coeffs = 1 << num_variables;

        let mut rng = ark_std::test_rng();
        let (leaf_hash_params, two_to_one_params) = merkle_tree::default_config::<F>(&mut rng);

        let mv_params = MultivariateParameters::<F>::new(num_variables);

        let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
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
        let statement = Statement {
            points: points.clone(),
            evaluations: points
                .iter()
                .map(|point| polynomial.evaluate(point))
                .collect(),
        };
        let mut evmfs_merlin = EVMFs::<F>::new();
        let committer = Committer::new(params.clone());
        let prover = Prover(params.clone());
        let verifier = Verifier::new(params.clone());
        let evm_witness = committer
            .evm_commit(&mut evmfs_merlin, polynomial.clone())
            .unwrap();
        let evm_proof = prover
            .evm_prove(&mut evmfs_merlin, statement.clone(), evm_witness)
            .unwrap();
        let mut evmfs_arthur = evmfs_merlin.to_arthur();
        // Return the untouched transcript
        let proof_transcript = evmfs_arthur.clone();
        assert!(verifier
            .evm_verify(&mut evmfs_arthur, &statement, &evm_proof)
            .is_ok());

        (params, proof_transcript, statement, evm_proof)
    }

    #[test]
    fn evm_test_whir() {
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
            for num_variables in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                evm_make_whir_things(
                                    num_variables,
                                    folding_factor,
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

    #[test]
    fn test_whir_single_poly() {
        let folding_factor = 1;
        let soundness_type = SoundnessType::UniqueDecoding;
        let fold_type = FoldType::ProverHelps;
        let num_points = 4;
        let pow_bits = 0;
        let num_variables = 3;
        evm_make_whir_things(
            num_variables as usize,
            folding_factor as usize,
            num_points,
            soundness_type,
            pow_bits,
            fold_type,
        );
    }

    #[test]
    fn test_serialize_full_proof() {
        let folding_factor = 2;
        let soundness_type = SoundnessType::ConjectureList;
        let fold_type = FoldType::ProverHelps;
        let num_points = 2;
        let pow_bits = 0;
        let num_variables = 3 * folding_factor;
        let proof = evm_make_whir_things(
            num_variables,
            folding_factor,
            num_points,
            soundness_type,
            pow_bits,
            fold_type,
        );

        let full_proof = FullEvmProof {
            whir_proof: convert_whir_proof::<PowStrategy, F>(proof.3).unwrap(),
            statement: proof.2,
            arthur: proof.1,
            config: proof.0,
        };
        let full_proof_json = serde_json::to_string_pretty(&full_proof).unwrap();
        let mut file = std::fs::File::create(format!(
            "proof_{}_{}_{}_{}_{}_{}.json",
            num_variables, folding_factor, num_points, soundness_type, pow_bits, fold_type
        ))
        .unwrap();
        file.write_all(full_proof_json.as_bytes()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use nimue::{plugins::pow::blake3::Blake3PoW, DefaultHash, IOPattern};

    use crate::crypto::fields::FieldBn256;
    use crate::crypto::merkle_tree::blake3 as merkle_tree;
    use crate::parameters::{FoldType, MultivariateParameters, SoundnessType, WhirParameters};
    use crate::poly_utils::coeffs::CoefficientList;
    use crate::poly_utils::MultilinearPoint;
    use crate::whir::Statement;
    use crate::whir::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier,
    };

    type MerkleConfig = merkle_tree::MerkleTreeParams<F>;
    type PowStrategy = Blake3PoW;
    type F = FieldBn256;

    fn make_whir_things(
        num_variables: usize,
        folding_factor: usize,
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

        let statement = Statement {
            points: points.clone(),
            evaluations: points
                .iter()
                .map(|point| polynomial.evaluate(point))
                .collect(),
        };

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial).unwrap();

        let prover = Prover(params.clone());

        let proof = prover
            .prove(&mut merlin, statement.clone(), witness)
            .unwrap();

        let verifier = Verifier::new(params);
        let mut arthur = io.to_arthur(merlin.transcript());
        assert!(verifier.verify(&mut arthur, &statement, &proof).is_ok());
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
            for num_variables in num_variables {
                for fold_type in fold_types {
                    for num_points in num_points {
                        for soundness_type in soundness_type {
                            for pow_bits in pow_bits {
                                make_whir_things(
                                    num_variables,
                                    folding_factor,
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

    #[test]
    fn test_serialize() {
        let path_sol_whir_test_data = std::env::current_dir().unwrap();
        let point = MultilinearPoint(vec![F::from(4242), F::from(2424)]);
        let statement = Statement {
            points: vec![point.clone(), point],
            evaluations: vec![F::from(42), F::from(42)],
        };
        let statement_json = serde_json::to_string(&statement).unwrap();
        let mut file =
            File::create(path_sol_whir_test_data.to_str().unwrap().to_owned() + "/statement.json")
                .expect("Failed at creating file");
        file.write_all(statement_json.as_bytes())
            .expect("Unable to write data");
    }

    #[test]
    fn test_whir_single_poly() {
        let folding_factor = 1;
        let soundness_type = SoundnessType::UniqueDecoding;
        let fold_type = FoldType::ProverHelps;
        let num_points = 4;
        let pow_bits = 0;
        let num_variables = 3;
        make_whir_things(
            num_variables as usize,
            folding_factor as usize,
            num_points,
            soundness_type,
            pow_bits,
            fold_type,
        );
    }
}
