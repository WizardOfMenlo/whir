use std::{
    fs::OpenOptions,
    time::{Duration, Instant},
};

use ark_serialize::CanonicalSerialize;
use nimue::{DefaultHash, IOPattern};
use whir::{
    crypto::{
        fields::Field192,
        merkle_tree::{blake3 as merkle_tree, HashCounter},
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
    #[arg(short = 'l', long, default_value = "128")]
    security_level: usize,

    #[arg(short = 'p', long, default_value = "106")]
    protocol_security_level: usize,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(long = "sk", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ConjectureList")]
    soundness_type: SoundnessType,
}

#[derive(Debug, Serialize)]
struct ProverOutput {
    security_level: usize,
    starting_rate: usize,
    num_variables: usize,
    repetitions: usize,
    folding_factor: usize,
    soundness_type: SoundnessType,
    // Whir
    whir_prover_time: Duration,
    whir_argument_size: usize,
    whir_prover_hashes: usize,

    // Whir LDT
    whir_ldt_prover_time: Duration,
    whir_ldt_argument_size: usize,
    whir_ldt_prover_hashes: usize,
}

type F = Field192;
type MerkleConfig = merkle_tree::MerkleTreeParams<F>;

fn main() {
    let args = Args::parse();

    let security_level = args.security_level;
    let protocol_security_level = args.protocol_security_level.min(security_level);
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;

    std::fs::create_dir_all("artifacts").unwrap();
    std::fs::create_dir_all("outputs").unwrap();

    let num_coeffs = 1 << num_variables;

    let mut rng = ark_std::test_rng();
    let (leaf_hash_params, two_to_one_params) =
        merkle_tree::default_config::<F>(&mut rng, folding_factor);

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = WhirParameters::<MerkleConfig> {
        protocol_security_level,
        security_level,
        folding_factor,
        leaf_hash_params,
        two_to_one_params,
        soundness_type,
        starting_log_inv_rate: starting_rate,
    };

    let polynomial = CoefficientList::new((0..num_coeffs).map(F::from).collect());

    let (whir_ldt_prover_time, whir_ldt_argument_size, whir_ldt_prover_hashes) = {
        // Run LDT
        use whir::whir_ldt::{
            committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
            whir_proof_size,
        };

        let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params.clone());

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

        // Serialize the proof
        let transcript = merlin.transcript().to_vec();
        let mut serialized_bytes = vec![];
        (transcript, proof)
            .serialize_compressed(&mut serialized_bytes)
            .unwrap();

        for i in 0..reps {
            std::fs::write(
                format!("artifacts/whir_ldt_proof{}", i),
                serialized_bytes.clone(),
            )
            .unwrap();
        }

        (
            whir_ldt_prover_time,
            whir_ldt_argument_size,
            whir_ldt_prover_hashes,
        )
    };

    let (whir_prover_time, whir_argument_size, whir_prover_hashes) = {
        // Run PCS
        use whir::poly_utils::MultilinearPoint;
        use whir::whir::{
            committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
            whir_proof_size,
        };

        let point = MultilinearPoint((0..num_variables as u32).map(F::from).collect());
        let eval = polynomial.evaluate(&point);
        let statement = Statement {
            points: vec![point],
            evaluations: vec![eval],
        };

        let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params);

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut merlin = io.to_merlin();

        let whir_prover_time = Instant::now();

        HashCounter::reset();

        let committer = Committer::new(params.clone());
        let witness = committer.commit(&mut merlin, polynomial.clone()).unwrap();

        let prover = Prover(params.clone());

        let proof = prover.prove(&mut merlin, statement, witness).unwrap();

        let whir_prover_time = whir_prover_time.elapsed();
        let whir_argument_size = whir_proof_size(merlin.transcript(), &proof);
        let whir_prover_hashes = HashCounter::get();

        // Serialize the proof
        let transcript = merlin.transcript().to_vec();
        let mut serialized_bytes = vec![];
        (transcript, proof)
            .serialize_compressed(&mut serialized_bytes)
            .unwrap();

        for i in 0..reps {
            std::fs::write(
                format!("artifacts/whir_proof{}", i),
                serialized_bytes.clone(),
            )
            .unwrap();
        }

        (whir_prover_time, whir_argument_size, whir_prover_hashes)
    };

    let output = ProverOutput {
        security_level,
        starting_rate,
        num_variables,
        repetitions: reps,
        folding_factor,
        soundness_type,
        // Whir
        whir_prover_time,
        whir_argument_size,
        whir_prover_hashes,

        // Whir LDT
        whir_ldt_prover_time,
        whir_ldt_argument_size,
        whir_ldt_prover_hashes,
    };

    let mut out_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("outputs/prover_output.json")
        .unwrap();
    use std::io::Write;
    writeln!(out_file, "{}", serde_json::to_string(&output).unwrap()).unwrap();
}
