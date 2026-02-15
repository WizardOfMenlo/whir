use std::{
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use serde::Serialize;
use whir::{
    algebra::{
        embedding::Basefield,
        fields,
        linear_form::{Evaluate, LinearForm, MultilinearEvaluation},
        polynomials::{CoefficientList, MultilinearPoint},
    },
    bits::Bits,
    cmdline_utils::{AvailableFields, AvailableHash},
    hash::HASH_COUNTER,
    parameters::{
        default_max_pow, FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType,
    },
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 'l', long, default_value = "128")]
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

    #[arg(short = 'i', long = "initfold", default_value = "4")]
    first_round_folding_factor: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ProvableList")]
    soundness_type: SoundnessType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks3")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    hash: AvailableHash,
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
    hash: AvailableHash,

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

fn main() {
    let mut args = Args::parse();
    let field = args.field;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    // Type reflection on field
    match field {
        AvailableFields::Goldilocks1 => run_whir::<fields::Field64>(&args),
        AvailableFields::Goldilocks2 => run_whir::<fields::Field64_2>(&args),
        AvailableFields::Goldilocks3 => run_whir::<fields::Field64_3>(&args),
        AvailableFields::Field128 => run_whir::<fields::Field128>(&args),
        AvailableFields::Field192 => run_whir::<fields::Field192>(&args),
        AvailableFields::Field256 => run_whir::<fields::Field256>(&args),
    }
}

#[allow(clippy::too_many_lines)]
fn run_whir<F>(args: &Args)
where
    F: FftField + CanonicalSerialize + Codec,
{
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let first_round_folding_factor = args.first_round_folding_factor;
    let soundness_type = args.soundness_type;

    std::fs::create_dir_all("outputs").unwrap();

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id: args.hash.hash_id(),
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
        use whir::protocols::whir::Config;

        let whir_params = ProtocolParameters {
            initial_statement: false,
            ..whir_params
        };
        let params = Config::<F>::new(mv_params, &whir_params);
        if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
            println!("WARN: more PoW bits required than what specified.");
        }

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Example at {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);

        let whir_ldt_prover_time = Instant::now();

        HASH_COUNTER.reset();

        let witness = params.commit(&mut prover_state, &[&polynomial]);

        let weights: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
        let evaluations: Vec<F> = Vec::new();
        let weight_refs = weights
            .iter()
            .map(|w| w.as_ref() as &dyn Evaluate<Basefield<F>>)
            .collect::<Vec<_>>();

        params.prove(
            &mut prover_state,
            &[&polynomial],
            &[&witness],
            &weight_refs,
            &evaluations,
        );

        let whir_ldt_prover_time = whir_ldt_prover_time.elapsed();
        let proof = prover_state.proof();
        let whir_ldt_argument_size = proof.narg_string.len() + proof.hints.len();
        let whir_ldt_prover_hashes = HASH_COUNTER.get();

        HASH_COUNTER.reset();
        let whir_ldt_verifier_time = Instant::now();
        let weight_refs = weights
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<F>)
            .collect::<Vec<_>>();
        for _ in 0..reps {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);

            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            params
                .verify(
                    &mut verifier_state,
                    &[&commitment],
                    &weight_refs,
                    &evaluations,
                )
                .unwrap();
        }

        let whir_ldt_verifier_time = whir_ldt_verifier_time.elapsed();
        let whir_ldt_verifier_hashes = HASH_COUNTER.get() / reps;

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
        use whir::protocols::whir::Config;

        let params = Config::<F>::new(mv_params, &whir_params);
        if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
            println!("WARN: more PoW bits required than what specified.");
        }

        let ds = DomainSeparator::protocol(&params)
            .session(&format!("Example at {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);

        let points: Vec<_> = (0..args.num_evaluations)
            .map(|i| MultilinearPoint(vec![F::from(i as u64); num_variables]))
            .collect();

        let mut weights: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
        let mut evaluations = Vec::new();

        for point in &points {
            let eval = polynomial.mixed_evaluate(&Basefield::new(), point);
            let weight = MultilinearEvaluation::new(point.0.clone());
            weights.push(Box::new(weight));
            evaluations.push(eval);
        }

        HASH_COUNTER.reset();
        let whir_prover_time = Instant::now();

        let witness = params.commit(&mut prover_state, &[&polynomial]);

        let weight_refs = weights
            .iter()
            .map(|w| w.as_ref() as &dyn Evaluate<Basefield<F>>)
            .collect::<Vec<_>>();
        params.prove(
            &mut prover_state,
            &[&polynomial],
            &[&witness],
            &weight_refs,
            &evaluations,
        );

        let whir_prover_time = whir_prover_time.elapsed();
        let proof = prover_state.proof();
        let whir_argument_size = proof.narg_string.len() + proof.hints.len();
        let whir_prover_hashes = HASH_COUNTER.get();

        HASH_COUNTER.reset();
        let whir_verifier_time = Instant::now();
        let weight_refs = weights
            .iter()
            .map(|w| w.as_ref() as &dyn LinearForm<F>)
            .collect::<Vec<_>>();
        for _ in 0..reps {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);

            let commitment = params.receive_commitment(&mut verifier_state).unwrap();
            params
                .verify(
                    &mut verifier_state,
                    &[&commitment],
                    &weight_refs,
                    &evaluations,
                )
                .unwrap();
        }

        let whir_verifier_time = whir_verifier_time.elapsed();
        let whir_verifier_hashes = HASH_COUNTER.get() / reps;

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
        hash: args.hash,

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
    writeln!(out_file, "{}", serde_json::to_string(&output).unwrap()).unwrap();
}
