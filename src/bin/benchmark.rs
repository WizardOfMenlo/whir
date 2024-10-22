use std::{
    fs::OpenOptions,
    time::{Duration, Instant},
};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
};
use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use nimue::{DefaultHash, IOPattern};
use nimue_pow::blake3::Blake3PoW;
use whir::{
    cmdline_utils::{AvailableFields, AvailableMerkle},
    crypto::{
        fields,
        merkle_tree::{self, HashCounter},
    },
    parameters::*,
    poly_utils::coeffs::CoefficientList,
    whir::Statement,
};

use serde::Serialize;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "100")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ConjectureList")]
    soundness_type: SoundnessType,

    #[arg(long = "fold_type", default_value = "ProverHelps")]
    fold_optimisation: FoldType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks2")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    merkle_tree: AvailableMerkle,
}

#[derive(Debug, Serialize)]
struct BenchmarkOutput {
    security_level: usize,
    pow_bits: usize,
    starting_rate: usize,
    num_variables: usize,
    repetitions: usize,
    folding_factor: usize,
    soundness_type: SoundnessType,
    field: AvailableFields,
    merkle_tree: AvailableMerkle,

    // Whir
    whir_evaluations: usize,
    whir_argument_size: usize,
    whir_prover_time: Duration,
    whir_prover_hashes: usize,
    whir_verifier_time: Duration,
    whir_verifier_hashes: usize,

    // Whir LDT
    whir_ldt_argument_size: usize,
    whir_ldt_prover_time: Duration,
    whir_ldt_prover_hashes: usize,
    whir_ldt_verifier_time: Duration,
    whir_ldt_verifier_hashes: usize,
}

type PowStrategy = Blake3PoW;

fn main() {
    let mut args = Args::parse();
    let field = args.field;
    let merkle = args.merkle_tree;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    let mut rng = ark_std::test_rng();

    match (field, merkle) {
        (AvailableFields::Goldilocks1, AvailableMerkle::Blake3) => {
            use fields::Field64 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks1, AvailableMerkle::Keccak256) => {
            use fields::Field64 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Blake3) => {
            use fields::Field64_2 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Keccak256) => {
            use fields::Field64_2 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Blake3) => {
            use fields::Field64_3 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Keccak256) => {
            use fields::Field64_3 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Blake3) => {
            use fields::Field128 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Keccak256) => {
            use fields::Field128 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Blake3) => {
            use fields::Field192 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Keccak256) => {
            use fields::Field192 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Blake3) => {
            use fields::Field256 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Keccak256) => {
            use fields::Field256 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }
    }
}

fn run_whir<F, MerkleConfig>(
    args: Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
{
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;
    let fold_optimisation = args.fold_optimisation;

    std::fs::create_dir_all("outputs").unwrap();

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
        security_level,
        pow_bits,
        folding_factor,
        leaf_hash_params,
        two_to_one_params,
        soundness_type,
        fold_optimisation,
        _pow_parameters: Default::default(),
        starting_log_inv_rate: starting_rate,
    };

    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );

    let (
        whir_ldt_prover_time,
        whir_ldt_argument_size,
        whir_ldt_prover_hashes,
        whir_ldt_verifier_time,
        whir_ldt_verifier_hashes,
    ) = {
        // Run LDT
        use whir::whir_ldt::{
            committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
            verifier::Verifier, whir_proof_size,
        };

        let params =
            WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params.clone());
        if !params.check_pow_bits() {
            println!("WARN: more PoW bits required than what specified.");
        }

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let whir_ldt_prover_time = Instant::now();

        HashCounter::reset();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial.clone()).unwrap();

        let prover = Prover(params.clone());

        let proof = prover.prove(&mut merlin, witness).unwrap();

        let whir_ldt_prover_time = whir_ldt_prover_time.elapsed();
        let whir_ldt_argument_size = whir_proof_size(merlin.transcript(), &proof);
        let whir_ldt_prover_hashes = HashCounter::get();

        // Just not to count that initial inversion (which could be precomputed)
        let verifier = Verifier::new(params);

        HashCounter::reset();
        let whir_ldt_verifier_time = Instant::now();
        for _ in 0..reps {
            let mut arthur = io.to_arthur(merlin.transcript());
            verifier.verify(&mut arthur, &proof).unwrap();
        }

        let whir_ldt_verifier_time = whir_ldt_verifier_time.elapsed();
        let whir_ldt_verifier_hashes = HashCounter::get() / reps;

        (
            whir_ldt_prover_time,
            whir_ldt_argument_size,
            whir_ldt_prover_hashes,
            whir_ldt_verifier_time,
            whir_ldt_verifier_hashes,
        )
    };

    let (
        whir_prover_time,
        whir_argument_size,
        whir_prover_hashes,
        whir_verifier_time,
        whir_verifier_hashes,
    ) = {
        // Run PCS
        use whir::poly_utils::MultilinearPoint;
        use whir::whir::{
            committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
            verifier::Verifier, whir_proof_size,
        };

        let params = WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params);
        if !params.check_pow_bits() {
            println!("WARN: more PoW bits required than what specified.");
        }

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let points: Vec<_> = (0..args.num_evaluations)
            .map(|i| MultilinearPoint(vec![F::from(i as u64); num_variables]))
            .collect();
        let evaluations = points
            .iter()
            .map(|point| polynomial.evaluate_at_extension(&point))
            .collect();
        let statement = Statement {
            points,
            evaluations,
        };

        HashCounter::reset();
        let whir_prover_time = Instant::now();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial.clone()).unwrap();

        let prover = Prover(params.clone());

        let proof = prover
            .prove(&mut merlin, statement.clone(), witness)
            .unwrap();

        let whir_prover_time = whir_prover_time.elapsed();
        let whir_argument_size = whir_proof_size(merlin.transcript(), &proof);
        let whir_prover_hashes = HashCounter::get();

        // Just not to count that initial inversion (which could be precomputed)
        let verifier = Verifier::new(params);

        HashCounter::reset();
        let whir_verifier_time = Instant::now();
        for _ in 0..reps {
            let mut arthur = io.to_arthur(merlin.transcript());
            verifier.verify(&mut arthur, &statement, &proof).unwrap();
        }

        let whir_verifier_time = whir_verifier_time.elapsed();
        let whir_verifier_hashes = HashCounter::get() / reps;

        (
            whir_prover_time,
            whir_argument_size,
            whir_prover_hashes,
            whir_verifier_time,
            whir_verifier_hashes,
        )
    };

    let output = BenchmarkOutput {
        security_level,
        pow_bits,
        starting_rate,
        num_variables,
        repetitions: reps,
        folding_factor,
        soundness_type,
        field: args.field,
        merkle_tree: args.merkle_tree,

        // Whir
        whir_evaluations: args.num_evaluations,
        whir_prover_time,
        whir_argument_size,
        whir_prover_hashes,
        whir_verifier_time,
        whir_verifier_hashes,

        // Whir LDT
        whir_ldt_prover_time,
        whir_ldt_argument_size,
        whir_ldt_prover_hashes,
        whir_ldt_verifier_time,
        whir_ldt_verifier_hashes,
    };

    let mut out_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("outputs/bench_output.json")
        .unwrap();
    use std::io::Write;
    writeln!(out_file, "{}", serde_json::to_string(&output).unwrap()).unwrap();
}
