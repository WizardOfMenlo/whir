use std::{
    fs::OpenOptions,
    time::{Duration, Instant},
};

use nimue::{DefaultHash, IOPattern};
use whir::{
    crypto::{
        fields::Field192,
        merkle_tree::{blake3 as merkle_tree, HashCounter},
    },
    parameters::*,
    poly_utils::coeffs::CoefficientList,
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
struct VerifierOutput {
    security_level: usize,
    starting_rate: usize,
    num_variables: usize,
    repetitions: usize,
    folding_factor: usize,
    soundness_type: SoundnessType,

    // Whir
    whir_verifier_time: Duration,
    whir_verifier_hashes: usize,

    // Whir LDT
    whir_ldt_verifier_time: Duration,
    whir_ldt_verifier_hashes: usize,
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

    let (whir_ldt_verifier_time, whir_ldt_verifier_hashes) = {
        // Run LDT
        use whir::whir_ldt::{
            iopattern::WhirIOPattern, parameters::WhirConfig, verifier::Verifier, WhirProof,
        };

        let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params.clone());

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut proofs = vec![];
        for i in 0..reps {
            let file_contents = std::fs::read(format!("artifacts/whir_ldt_proof{}", i)).unwrap();
            let (transcript, proof): (Vec<u8>, WhirProof<MerkleConfig>) =
                ark_serialize::CanonicalDeserialize::deserialize_compressed(
                    &mut &file_contents[..],
                )
                .unwrap();
            proofs.push((transcript, proof));
        }

        let verifier = Verifier::new(params);

        HashCounter::reset();
        let stir_verifier_time = Instant::now();
        for (transcript, proof) in proofs {
            let mut arthur = io.to_arthur(&transcript);
            verifier.verify(&mut arthur, &proof).unwrap();
        }
        let whir_ldt_verifier_time = stir_verifier_time.elapsed();
        let whir_ldt_verifier_hashes = HashCounter::get() / reps;

        (whir_ldt_verifier_time, whir_ldt_verifier_hashes)
    };

    let (whir_verifier_time, whir_verifier_hashes) = {
        use whir::poly_utils::MultilinearPoint;
        use whir::whir::{
            iopattern::WhirIOPattern, parameters::WhirConfig, verifier::Verifier, Statement,
            WhirProof,
        };

        let point = MultilinearPoint((0..num_variables as u32).map(F::from).collect());
        let polynomial = CoefficientList::new((0..(1 << num_variables)).map(F::from).collect());
        let eval = polynomial.evaluate(&point);
        let statement = Statement {
            points: vec![point],
            evaluations: vec![eval],
        };

        let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params.clone());

        let io = IOPattern::<DefaultHash>::new("🌪️")
            .commit_statement(&params)
            .add_whir_proof(&params)
            .clone();

        let mut proofs = vec![];
        for i in 0..reps {
            let file_contents = std::fs::read(format!("artifacts/whir_proof{}", i)).unwrap();
            let (transcript, proof): (Vec<u8>, WhirProof<MerkleConfig>) =
                ark_serialize::CanonicalDeserialize::deserialize_compressed(
                    &mut &file_contents[..],
                )
                .unwrap();
            proofs.push((transcript, proof));
        }

        let verifier = Verifier::new(params);

        HashCounter::reset();
        let stir_verifier_time = Instant::now();
        for (transcript, proof) in proofs {
            let mut arthur = io.to_arthur(&transcript);
            verifier.verify(&mut arthur, &statement, &proof).unwrap();
        }
        let whir_verifier_time = stir_verifier_time.elapsed();
        let whir_verifier_hashes = HashCounter::get() / reps;

        (whir_verifier_time, whir_verifier_hashes)
    };

    let output = VerifierOutput {
        security_level,
        starting_rate,
        num_variables,
        repetitions: reps,
        folding_factor,
        soundness_type,
        whir_verifier_time,
        whir_verifier_hashes,
        whir_ldt_verifier_time,
        whir_ldt_verifier_hashes,
    };

    let mut out_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("outputs/verifier_output.json")
        .unwrap();
    use std::io::Write;
    writeln!(out_file, "{}", serde_json::to_string(&output).unwrap()).unwrap();
}
