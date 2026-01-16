use std::{sync::Arc, time::Instant};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
};
use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use spongefish::{domain_separator, session, Codec};
use whir::{
    cmdline_utils::{AvailableFields, AvailableMerkle, WhirType},
    crypto::{
        fields,
        merkle_tree::{
            blake3::Blake3MerkleTreeParams, keccak::KeccakMerkleTreeParams, HashCounter,
        },
    },
    ntt::{RSDefault, ReedSolomon},
    parameters::{
        default_max_pow, DeduplicationStrategy, FoldingFactor, MerkleProofStrategy,
        MultivariateParameters, ProtocolParameters, SoundnessType,
    },
    poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    transcript::{codecs::Empty, ProverMessage, ProverState, VerifierState},
    whir::{
        committer::CommitmentReader,
        statement::{Statement, Weights},
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 't', long = "type", default_value = "PCS")]
    protocol_type: WhirType,

    #[arg(short = 'l', long, default_value = "128")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(long = "linear_constraints", default_value = "0")]
    num_linear_constraints: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ProvableList")]
    soundness_type: SoundnessType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    merkle_tree: AvailableMerkle,
}

fn main() {
    let mut args = Args::parse();
    let field = args.field;
    let merkle = args.merkle_tree;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    runner(&args, field, merkle);
}

fn runner(args: &Args, field: AvailableFields, merkle: AvailableMerkle) {
    // Type reflection on field
    match field {
        AvailableFields::Goldilocks1 => {
            use fields::Field64 as F;
            runner_merkle::<F>(args, merkle);
        }
        AvailableFields::Goldilocks2 => {
            use fields::Field64_2 as F;
            runner_merkle::<F>(args, merkle);
        }
        AvailableFields::Goldilocks3 => {
            use fields::Field64_3 as F;
            runner_merkle::<F>(args, merkle);
        }
        AvailableFields::Field128 => {
            use fields::Field128 as F;
            runner_merkle::<F>(args, merkle);
        }
        AvailableFields::Field192 => {
            use fields::Field192 as F;
            runner_merkle::<F>(args, merkle);
        }
        AvailableFields::Field256 => {
            use fields::Field256 as F;
            runner_merkle::<F>(args, merkle);
        }
    }
}

fn runner_merkle<F: FftField + CanonicalSerialize + Codec>(args: &Args, merkle: AvailableMerkle) {
    let reed_solomon = Arc::new(RSDefault);
    let basefield_reed_solomon = reed_solomon.clone();

    // Type reflection on merkle
    match merkle {
        AvailableMerkle::Blake3 => {
            run_whir::<F, Blake3MerkleTreeParams<F>>(args, reed_solomon, basefield_reed_solomon);
        }
        AvailableMerkle::Keccak256 => {
            run_whir::<F, KeccakMerkleTreeParams<F>>(args, reed_solomon, basefield_reed_solomon);
        }
    }
}

fn run_whir<F, MerkleConfig>(
    args: &Args,
    reed_solomon: Arc<dyn ReedSolomon<F>>,
    basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
) where
    F: FftField + CanonicalSerialize + Codec,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]> + ProverMessage,
    MerkleConfig::LeafHash: CRHScheme<Parameters = ()>,
    MerkleConfig::TwoToOneHash: TwoToOneCRHScheme<Parameters = ()>,
{
    match args.protocol_type {
        WhirType::PCS => {
            run_whir_pcs::<F, MerkleConfig>(args, reed_solomon, basefield_reed_solomon);
        }
        WhirType::LDT => {
            run_whir_as_ldt::<F, MerkleConfig>(args, reed_solomon, basefield_reed_solomon);
        }
    }
}

fn run_whir_as_ldt<F, MerkleConfig>(
    args: &Args,
    reed_solomon: Arc<dyn ReedSolomon<F>>,
    basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
) where
    F: FftField + CanonicalSerialize + Codec,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]> + ProverMessage,
    MerkleConfig::LeafHash: CRHScheme<Parameters = ()>,
    MerkleConfig::TwoToOneHash: TwoToOneCRHScheme<Parameters = ()>,
{
    use whir::whir::{
        committer::CommitmentWriter, parameters::WhirConfig, prover::Prover, verifier::Verifier,
    };

    // Runs as a LDT
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;

    if args.num_evaluations > 1 {
        println!("Warning: running as LDT but a number of evaluations to be proven was specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters::<MerkleConfig> {
        initial_statement: false,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        leaf_hash_params: (),
        two_to_one_params: (),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        deduplication_strategy: DeduplicationStrategy::Enabled,
        merkle_proof_strategy: MerkleProofStrategy::Compressed,
    };

    let params = WhirConfig::<F, MerkleConfig>::new(
        reed_solomon,
        basefield_reed_solomon,
        mv_params,
        whir_params,
    );

    let ds = domain_separator!("🌪️")
        .session(session!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::from(ds.std_prover());

    println!("=========================================");
    println!("Whir (LDT) 🌪️");
    println!("Field: {:?} and MT: {:?}", args.field, args.merkle_tree);
    println!("{params}");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );

    let whir_prover_time = Instant::now();

    let committer = CommitmentWriter::new(params.clone());
    let witness = committer.commit(prover_state.inner_mut(), &polynomial);

    let prover = Prover::new(params.clone());

    let statement = Statement::new(num_variables);
    prover.prove(&mut prover_state, statement.clone(), witness);

    // Serialize proof
    let proof = prover_state.proof();
    let proof_size = proof.narg_string.len() + proof.hints.len();
    println!("Prover time: {:.1?}", whir_prover_time.elapsed());
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);

    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        let parsed_commitment = commitment_reader
            .parse_commitment(verifier_state.inner_mut())
            .unwrap();
        verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement)
            .unwrap();
    }
    dbg!(whir_verifier_time.elapsed() / reps as u32);
    dbg!(HashCounter::get() as f64 / reps as f64);
}

#[allow(clippy::too_many_lines)]
fn run_whir_pcs<F, MerkleConfig>(
    args: &Args,
    reed_solomon: Arc<dyn ReedSolomon<F>>,
    basefield_reed_solomon: Arc<dyn ReedSolomon<F::BasePrimeField>>,
) where
    F: FftField + CanonicalSerialize + Codec,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]> + ProverMessage,
    MerkleConfig::LeafHash: CRHScheme<Parameters = ()>,
    MerkleConfig::TwoToOneHash: TwoToOneCRHScheme<Parameters = ()>,
{
    use whir::whir::{
        committer::CommitmentWriter, parameters::WhirConfig, prover::Prover, statement::Statement,
        verifier::Verifier,
    };

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters::<MerkleConfig> {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        leaf_hash_params: (),
        two_to_one_params: (),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        deduplication_strategy: DeduplicationStrategy::Enabled,
        merkle_proof_strategy: MerkleProofStrategy::Compressed,
    };

    let params = WhirConfig::<F, MerkleConfig>::new(
        reed_solomon,
        basefield_reed_solomon,
        mv_params,
        whir_params,
    );

    let ds = domain_separator!("🌪️")
        .session(session!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::from(ds.std_prover());

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and MT: {:?}", args.field, args.merkle_tree);
    println!("{params}");
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );
    let whir_prover_time = Instant::now();

    let committer = CommitmentWriter::new(params.clone());
    let witness = committer.commit(prover_state.inner_mut(), &polynomial);

    let mut statement: Statement<F> = Statement::<F>::new(num_variables);

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![F::from(x as u64); num_variables]))
        .collect();

    for point in &points {
        let eval = polynomial.evaluate_at_extension(point);
        let weights = Weights::evaluation(point.clone());
        statement.add_constraint(weights, eval);
    }

    // Linear constraint
    for _ in 0..num_linear_constraints {
        let input = CoefficientList::new((0..num_coeffs).map(F::from).collect());
        let input: EvaluationsList<F> = input.clone().into();

        let linear_claim_weight = Weights::linear(input.clone());
        let poly = EvaluationsList::from(polynomial.clone().to_extension());

        let sum = linear_claim_weight.weighted_sum(&poly);
        statement.add_constraint(linear_claim_weight, sum);
    }

    let prover = Prover::new(params.clone());

    prover.prove(&mut prover_state, statement.clone(), witness);

    let proof = prover_state.proof();
    println!("Prover time: {:.1?}", whir_prover_time.elapsed());
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state =
            VerifierState::from(ds.std_verifier(&proof.narg_string), &proof.hints);

        let parsed_commitment = commitment_reader
            .parse_commitment(verifier_state.inner_mut())
            .unwrap();
        verifier
            .verify(&mut verifier_state, &parsed_commitment, &statement)
            .unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HashCounter::get() as f64 / reps as f64) / 1000.0
    );
}
