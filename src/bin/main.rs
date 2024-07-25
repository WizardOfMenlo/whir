use std::time::Instant;

use nimue::{DefaultHash, IOPattern};
use whir::{
    crypto::{
        fields::Field192,
        merkle_tree::{blake3 as merkle_tree, HashCounter},
    },
    parameters::*,
    poly_utils::coeffs::CoefficientList,
    whir_ldt::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier, whir_proof_size,
    },
};

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

type F = Field192;
type MerkleConfig = merkle_tree::MerkleTreeParams<F>;

fn main() {
    // Runs as a LDT
    let args = Args::parse();

    let security_level = args.security_level;
    let protocol_security_level = args.protocol_security_level.min(security_level);
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;

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

    let params = WhirConfig::<F, MerkleConfig>::new(mv_params, whir_params);

    let polynomial = CoefficientList::new((0..num_coeffs).map(F::from).collect());

    let io = IOPattern::<DefaultHash>::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params)
        .clone();

    let mut merlin = io.to_merlin();

    println!("=========================================");
    println!("Whir 🌪️");
    println!("{}", params);

    let whir_prover_time = Instant::now();

    let committer = Committer::new(params.clone());
    let witness = committer.commit(&mut merlin, polynomial).unwrap();

    let prover = Prover(params.clone());

    let proof = prover.prove(&mut merlin, witness).unwrap();

    dbg!(whir_prover_time.elapsed());
    dbg!(whir_proof_size(merlin.transcript(), &proof));

    // Just not to count that initial inversion (which could be precomputed)
    let verifier = Verifier::new(params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut arthur = io.to_arthur(merlin.transcript());
        verifier.verify(&mut arthur, &proof).unwrap();
    }
    dbg!(whir_verifier_time.elapsed() / reps as u32);
    dbg!(HashCounter::get() as f64 / reps as f64);
}
