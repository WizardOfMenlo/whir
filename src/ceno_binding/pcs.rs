use super::{Error, PolynomialCommitmentScheme};
use crate::crypto::merkle_tree::blake3::{self as mt, MerkleTreeParams};
use crate::parameters::{
    default_max_pow, FoldType, MultivariateParameters, SoundnessType, WhirParameters,
};
use crate::poly_utils::{coeffs::CoefficientList, MultilinearPoint};
use crate::whir::{
    committer::{Committer, Witness},
    iopattern::WhirIOPattern,
    parameters::WhirConfig,
    prover::Prover,
    verifier::Verifier,
    Statement, WhirProof,
};

use ark_crypto_primitives::merkle_tree::{Config, MerkleTree};
use ark_ff::FftField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use nimue::{DefaultHash, IOPattern, Merlin};
use nimue_pow::blake3::Blake3PoW;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "parallel")]
use rayon::slice::ParallelSlice;
use serde::ser::SerializeStruct;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Whir<E>(PhantomData<E>);

type MerkleConfig<E> = MerkleTreeParams<E>;
type PowStrategy = Blake3PoW;
type WhirPCSConfig<E> = WhirConfig<E, MerkleConfig<E>, PowStrategy>;

// Wrapper for WhirConfig
pub struct WhirConfigWrapper<E: FftField> {
    inner: WhirConfig<E, MerkleConfig<E>, PowStrategy>,
}

impl<E: FftField> Serialize for WhirConfigWrapper<E> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("WhirConfigWrapper", 17)?;
        s.serialize_field(
            "num_variables",
            &(self.inner.mv_parameters.num_variables as u32),
        )?;
        s.serialize_field("initial_statement", &self.inner.initial_statement)?;
        s.serialize_field("starting_log_inv_rate", &self.inner.starting_log_inv_rate)?;
        s.serialize_field("folding_factor", &self.inner.folding_factor)?;
        s.serialize_field("soundness_type", &self.inner.soundness_type)?;
        s.serialize_field("security_level", &self.inner.security_level)?;
        s.serialize_field("pow_bits", &self.inner.max_pow_bits)?;
        s.serialize_field("fold_optimisation", &self.inner.fold_optimisation)?;

        s.end()
    }
}

impl<'de, E: FftField> Deserialize<'de> for WhirConfigWrapper<E> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor<E> {
            marker: PhantomData<E>,
        }

        impl<'de, E: FftField> serde::de::Visitor<'de> for Visitor<E> {
            type Value = WhirConfig<E, MerkleConfig<E>, PowStrategy>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct WhirConfigWrapper")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut num_variables = None;
                let mut soundness_type = None;
                let mut security_level = None;
                let mut pow_bits = None;
                let mut initial_statement = None;
                let mut starting_log_inv_rate = None;
                let mut folding_factor = None;
                let mut fold_optimisation = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "num_variables" => {
                            if num_variables.is_some() {
                                return Err(serde::de::Error::duplicate_field("num_variables"));
                            }
                            num_variables = Some(map.next_value()?);
                        }
                        "soundness_type" => {
                            if soundness_type.is_some() {
                                return Err(serde::de::Error::duplicate_field("soundness_type"));
                            }
                            soundness_type = Some(map.next_value()?);
                        }
                        "security_level" => {
                            if security_level.is_some() {
                                return Err(serde::de::Error::duplicate_field("security_level"));
                            }
                            security_level = Some(map.next_value()?);
                        }
                        "pow_bits" => {
                            if pow_bits.is_some() {
                                return Err(serde::de::Error::duplicate_field("pow_bits"));
                            }
                            pow_bits = Some(map.next_value()?);
                        }
                        "initial_statement" => {
                            if initial_statement.is_some() {
                                return Err(serde::de::Error::duplicate_field("initial_statement"));
                            }
                            initial_statement = Some(map.next_value()?);
                        }
                        "starting_log_inv_rate" => {
                            if starting_log_inv_rate.is_some() {
                                return Err(serde::de::Error::duplicate_field(
                                    "starting_log_inv_rate",
                                ));
                            }
                            starting_log_inv_rate = Some(map.next_value()?);
                        }
                        "folding_factor" => {
                            if folding_factor.is_some() {
                                return Err(serde::de::Error::duplicate_field("folding_factor"));
                            }
                            folding_factor = Some(map.next_value()?);
                        }
                        "fold_optimisation" => {
                            if fold_optimisation.is_some() {
                                return Err(serde::de::Error::duplicate_field("fold_optimisation"));
                            }
                            fold_optimisation = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(serde::de::Error::unknown_field(
                                key,
                                &[
                                    "num_variables",
                                    "soundness_type",
                                    "security_level",
                                    "pow_bits",
                                    "initial_statement",
                                    "starting_log_inv_rate",
                                    "folding_factor",
                                    "fold_optimisation",
                                ],
                            ));
                        }
                    }
                }

                let num_variables = num_variables
                    .ok_or_else(|| serde::de::Error::missing_field("num_variables"))?;
                let soundness_type = soundness_type
                    .ok_or_else(|| serde::de::Error::missing_field("soundness_type"))?;
                let security_level = security_level
                    .ok_or_else(|| serde::de::Error::missing_field("security_level"))?;
                let pow_bits =
                    pow_bits.ok_or_else(|| serde::de::Error::missing_field("pow_bits"))?;
                let initial_statement = initial_statement
                    .ok_or_else(|| serde::de::Error::missing_field("initial_statement"))?;
                let starting_log_inv_rate = starting_log_inv_rate
                    .ok_or_else(|| serde::de::Error::missing_field("starting_log_inv_rate"))?;
                let folding_factor = folding_factor
                    .ok_or_else(|| serde::de::Error::missing_field("folding_factor"))?;
                let fold_optimisation = fold_optimisation
                    .ok_or_else(|| serde::de::Error::missing_field("fold_optimisation"))?;

                let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
                let (leaf_hash_params, two_to_one_params) = mt::default_config::<E>(&mut rng);
                Ok(WhirConfig::new(
                    MultivariateParameters::new(num_variables),
                    WhirParameters {
                        initial_statement,
                        starting_log_inv_rate,
                        folding_factor,
                        soundness_type,
                        security_level,
                        pow_bits,
                        fold_optimisation,
                        _pow_parameters: PhantomData::<PowStrategy>,
                        // Merkle tree parameters
                        leaf_hash_params,
                        two_to_one_params,
                    },
                ))
            }
        }

        let config = deserializer.deserialize_struct(
            "WhirConfigWrapper",
            &[
                "num_variables",
                "soundness_type",
                "security_level",
                "pow_bits",
                "initial_statement",
                "starting_log_inv_rate",
                "folding_factor",
                "fold_optimisation",
            ],
            Visitor {
                marker: PhantomData,
            },
        )?;

        Ok(WhirConfigWrapper { inner: config })
    }
}

impl<E: FftField> Debug for WhirConfigWrapper<E> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("WhirConfigWrapper")
    }
}

impl<E: FftField> Clone for WhirConfigWrapper<E> {
    fn clone(&self) -> Self {
        WhirConfigWrapper {
            inner: self.inner.clone(),
        }
    }
}

// Wrapper for Witness
pub struct WitnessWrapper<F: FftField> {
    inner: Witness<F, MerkleTreeParams<F>>,
}

impl<F: FftField> Serialize for WitnessWrapper<F>
where
    F: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("WitnessWrapper", 3)?;
        s.serialize_field("polynomial", &self.inner.polynomial)?;
        s.serialize_field("merkle_leaves", &self.inner.merkle_leaves)?;
        s.serialize_field("ood_points", &self.inner.ood_points)?;
        s.serialize_field("ood_answers", &self.inner.ood_answers)?;
        s.serialize_field("tree_height", &self.inner.merkle_tree.height())?;
        s.end()
    }
}

impl<'de, F> Deserialize<'de> for WitnessWrapper<F>
where
    F: Deserialize<'de> + FftField + CanonicalDeserialize + CanonicalSerialize,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct Visitor<F> {
            marker: PhantomData<F>,
        }

        impl<'de, F: FftField> serde::de::Visitor<'de> for Visitor<F>
        where
            F: FftField + Deserialize<'de> + CanonicalDeserialize + CanonicalSerialize,
        {
            type Value = WitnessWrapper<F>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct WitnessWrapper")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut polynomial = None;
                let mut merkle_leaves = None;
                let mut ood_points = None;
                let mut ood_answers = None;
                let mut tree_height = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "polynomial" => {
                            if polynomial.is_some() {
                                return Err(serde::de::Error::duplicate_field("polynomial"));
                            }
                            polynomial = Some(map.next_value()?);
                        }
                        "merkle_leaves" => {
                            if merkle_leaves.is_some() {
                                return Err(serde::de::Error::duplicate_field("merkle_leaves"));
                            }
                            merkle_leaves = Some(map.next_value()?);
                        }
                        "ood_points" => {
                            if ood_points.is_some() {
                                return Err(serde::de::Error::duplicate_field("ood_points"));
                            }
                            ood_points = Some(map.next_value()?);
                        }
                        "ood_answers" => {
                            if ood_answers.is_some() {
                                return Err(serde::de::Error::duplicate_field("ood_answers"));
                            }
                            ood_answers = Some(map.next_value()?);
                        }
                        "tree_height" => {
                            if tree_height.is_some() {
                                return Err(serde::de::Error::duplicate_field("tree_height"));
                            }
                            tree_height = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(serde::de::Error::unknown_field(
                                key,
                                &[
                                    "polynomial",
                                    "merkle_leaves",
                                    "ood_points",
                                    "ood_answers",
                                    "tree_height",
                                ],
                            ));
                        }
                    }
                }

                let polynomial =
                    polynomial.ok_or_else(|| serde::de::Error::missing_field("polynomial"))?;
                let merkle_leaves: Vec<F> = merkle_leaves
                    .ok_or_else(|| serde::de::Error::missing_field("merkle_leaves"))?;
                let ood_points =
                    ood_points.ok_or_else(|| serde::de::Error::missing_field("ood_points"))?;
                let ood_answers =
                    ood_answers.ok_or_else(|| serde::de::Error::missing_field("ood_answers"))?;

                let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
                let (leaf_hash_param, two_to_one_hash_param) = mt::default_config::<F>(&mut rng);

                let tree_height: usize =
                    tree_height.ok_or_else(|| serde::de::Error::missing_field("tree_height"))?;
                let leaf_node_size = 1 << (tree_height - 1);
                let fold_size = merkle_leaves.len() / leaf_node_size;
                #[cfg(not(feature = "parallel"))]
                let leafs_iter = merkle_leaves.chunks_exact(fold_size);
                #[cfg(feature = "parallel")]
                let leafs_iter = merkle_leaves.par_chunks_exact(fold_size);
                let merkle_tree =
                    MerkleTree::new(&leaf_hash_param, &two_to_one_hash_param, leafs_iter).map_err(
                        |_| serde::de::Error::custom("Failed to construct the merkle tree"),
                    )?;

                Ok(WitnessWrapper {
                    inner: Witness {
                        polynomial,
                        merkle_tree,
                        merkle_leaves,
                        ood_points,
                        ood_answers,
                    },
                })
            }
        }

        deserializer.deserialize_struct(
            "WitnessWrapper",
            &[
                "polynomial",
                "merkle_leaves",
                "ood_points",
                "ood_answers",
                "tree_height",
            ],
            Visitor {
                marker: PhantomData,
            },
        )
    }
}

impl<F: FftField> Debug for WitnessWrapper<F> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str("WitnessWrapper")
    }
}

impl<F: FftField> Clone for WitnessWrapper<F> {
    fn clone(&self) -> Self {
        WitnessWrapper {
            inner: self.inner.clone(),
        }
    }
}

impl<E> PolynomialCommitmentScheme<E> for Whir<E>
where
    E: FftField + CanonicalSerialize + CanonicalDeserialize + Serialize + DeserializeOwned + Debug,
    E::BasePrimeField: Serialize + DeserializeOwned + Debug,
{
    type Param = WhirConfigWrapper<E>;
    type CommitmentWithData = WitnessWrapper<E>;
    type Proof = WhirProof<MerkleTreeParams<E>, E>;
    type Poly = CoefficientList<E::BasePrimeField>;
    type Transcript = Merlin<DefaultHash>;

    fn setup(poly_size: usize) -> Self::Param {
        let mv_params = MultivariateParameters::<E>::new(poly_size);
        let starting_rate = 1;
        let pow_bits = default_max_pow(poly_size, starting_rate);
        let mut rng = ChaCha8Rng::from_seed([0u8; 32]);

        let (leaf_hash_params, two_to_one_params) = mt::default_config::<E>(&mut rng);

        let whir_params = WhirParameters::<MerkleTreeParams<E>, PowStrategy> {
            initial_statement: true,
            security_level: 100,
            pow_bits,
            folding_factor: 4,
            leaf_hash_params,
            two_to_one_params,
            soundness_type: SoundnessType::ConjectureList,
            fold_optimisation: FoldType::ProverHelps,
            _pow_parameters: Default::default(),
            starting_log_inv_rate: starting_rate,
        };

        WhirConfigWrapper {
            inner: WhirConfig::<E, MerkleConfig<E>, PowStrategy>::new(mv_params, whir_params),
        }
    }

    fn commit_and_write(
        pp: &Self::Param,
        poly: &Self::Poly,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::CommitmentWithData, Error> {
        let committer = Committer::new(pp.inner.clone());
        let witness = committer.commit(transcript, poly.clone())?;

        Ok(WitnessWrapper { inner: witness })
    }

    fn batch_commit(
        _pp: &Self::Param,
        _polys: &[Self::Poly],
    ) -> Result<Self::CommitmentWithData, Error> {
        todo!()
    }

    fn open(
        pp: &Self::Param,
        witness: Self::CommitmentWithData,
        point: &[E],
        eval: &E,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::Proof, Error> {
        let prover = Prover(pp.inner.clone());
        let statement = Statement {
            points: vec![MultilinearPoint(point.to_vec())],
            evaluations: vec![eval.clone()],
        };

        let proof = prover.prove(transcript, statement, witness.inner)?;
        Ok(proof)
    }

    fn batch_open(
        _pp: &Self::Param,
        _polys: &[Self::Poly],
        _comm: Self::CommitmentWithData,
        _point: &[E],
        _evals: &[E],
        _transcript: &mut Self::Transcript,
    ) -> Result<Self::Proof, Error> {
        todo!()
    }

    fn verify(
        vp: &Self::Param,
        point: &[E],
        eval: &E,
        proof: &Self::Proof,
        transcript: &Self::Transcript,
    ) -> Result<(), Error> {
        let reps = 1000;
        let verifier = Verifier::new(vp.inner.clone());
        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&vp.inner)
            .add_whir_proof(&vp.inner);

        let statement = Statement {
            points: vec![MultilinearPoint(point.to_vec())],
            evaluations: vec![eval.clone()],
        };

        for _ in 0..reps {
            let mut arthur = io.to_arthur(transcript.transcript());
            verifier.verify(&mut arthur, &statement, proof)?;
        }
        Ok(())
    }

    fn batch_verify(
        _vp: &Self::Param,
        _point: &[E],
        _evals: &[E],
        _proof: &Self::Proof,
        _transcript: &mut Self::Transcript,
    ) -> Result<(), Error> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;
    use rand::Rng;

    use super::*;
    use crate::crypto::fields::Field64_2 as F;

    #[test]
    fn single_point_verify() {
        let poly_size = 10;
        let num_coeffs = 1 << poly_size;
        let pp = Whir::<F>::setup(poly_size);

        let poly = CoefficientList::new(
            (0..num_coeffs)
                .map(<F as Field>::BasePrimeField::from)
                .collect(),
        );

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&pp)
            .add_whir_proof(&pp);
        let mut merlin = io.to_merlin();

        let witness = Whir::<F>::commit_and_write(&pp, &poly, &mut merlin).unwrap();

        let mut rng = rand::thread_rng();
        let point: Vec<F> = (0..poly_size).map(|_| F::from(rng.gen::<u64>())).collect();
        let eval = poly.evaluate_at_extension(&MultilinearPoint(point.clone()));

        let proof = Whir::<F>::open(&pp, witness, &point, &eval, &mut merlin).unwrap();
        Whir::<F>::verify(&pp, &point, &eval, &proof, &merlin).unwrap();
    }
}
