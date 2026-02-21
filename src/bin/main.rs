use std::{borrow::Cow, time::Instant};

use ark_ff::{FftField, Field, PrimeField};
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use whir::{
    algebra::{
        embedding::{Basefield, Identity},
        fields,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
        MultilinearPoint,
    },
    bits::Bits,
    cmdline_utils::{AvailableFields, AvailableHash, WhirType},
    hash::HASH_COUNTER,
    parameters::{
        default_max_pow, FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType,
    },
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
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
    hash: AvailableHash,

    #[arg(long = "zk")]
    zk: bool,
}

fn main() {
    let mut args = Args::parse();
    let field = args.field;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    if let Err(err) = validate_args(&args) {
        eprintln!("Error: {err}");
        std::process::exit(2);
    }

    runner(&args, field);
}

fn validate_args(args: &Args) -> Result<(), String> {
    if args.zk && !matches!(args.protocol_type, WhirType::PCS) {
        return Err("--zk is only supported with --type PCS".into());
    }
    Ok(())
}

fn runner(args: &Args, field: AvailableFields) {
    match (args.protocol_type, args.zk, field) {
        // zkWHIR is currently pinned to identity embedding over a single field.
        (WhirType::PCS, true, AvailableFields::Goldilocks1) => {
            run_whir_pcs_zk::<fields::Field64>(args);
        }
        (WhirType::PCS, true, AvailableFields::Field128) => {
            run_whir_pcs_zk::<fields::Field128>(args);
        }
        (WhirType::PCS, true, AvailableFields::Field192) => {
            run_whir_pcs_zk::<fields::Field192>(args);
        }
        (WhirType::PCS, true, AvailableFields::Field256) => {
            run_whir_pcs_zk::<fields::Field256>(args);
        }
        (WhirType::PCS, true, _) => {
            eprintln!(
                "Error: --zk supports only single-field configurations (Identity embedding)."
            );
        }
        (WhirType::PCS | WhirType::LDT, false, f) => run_whir_for_field(args, f),
        (WhirType::LDT, true, _) => unreachable!("validated earlier"),
    }
}

fn run_whir_for_field(args: &Args, field: AvailableFields) {
    match field {
        AvailableFields::Goldilocks1 => run_whir::<fields::Field64>(args),
        AvailableFields::Goldilocks2 => run_whir::<fields::Field64_2>(args),
        AvailableFields::Goldilocks3 => run_whir::<fields::Field64_3>(args),
        AvailableFields::Field128 => run_whir::<fields::Field128>(args),
        AvailableFields::Field192 => run_whir::<fields::Field192>(args),
        AvailableFields::Field256 => run_whir::<fields::Field256>(args),
    }
}

fn run_whir<F>(args: &Args)
where
    F: FftField + CanonicalSerialize + Codec,
{
    match args.protocol_type {
        WhirType::PCS => {
            run_whir_pcs::<F>(args);
        }
        WhirType::LDT => {
            run_whir_as_ldt::<F>(args);
        }
    }
}

fn run_whir_as_ldt<F>(args: &Args)
where
    F: FftField + CanonicalSerialize + Codec,
{
    use whir::protocols::whir::Config;

    // Runs as a LDT
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let soundness_type = args.soundness_type;
    let hash_id = args.hash.hash_id();

    if args.num_evaluations > 1 {
        println!("Warning: running as LDT but a number of evaluations to be proven was specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters {
        initial_statement: false,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        soundness_type,
        starting_log_inv_rate: starting_rate,
        batch_size: 1,
        hash_id,
    };

    let params = Config::<F>::new(mv_params, &whir_params);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (LDT) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
        println!("WARN: more PoW bits required than what specified.");
    }

    let vector = (0..num_coeffs)
        .map(<F as Field>::BasePrimeField::from)
        .collect::<Vec<_>>();

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &[&vector]);
    let whir_commit_time = whir_commit_time.elapsed();

    let whir_prove_time = Instant::now();
    params.prove(
        &mut prover_state,
        vec![Cow::from(vector)],
        vec![Cow::Owned(witness)],
        &[],
        Cow::Owned(vec![]),
    );
    let whir_prove_time = whir_prove_time.elapsed();

    // Serialize proof
    let proof = prover_state.proof();
    let proof_size = proof.narg_string.len() + proof.hints.len();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!("Proof size: {:.1} KiB", proof_size as f64 / 1024.0);

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();
        params
            .verify(&mut verifier_state, &[&commitment], &[], &[])
            .unwrap();
    }
    dbg!(whir_verifier_time.elapsed() / reps as u32);
    dbg!(HASH_COUNTER.get() as f64 / reps as f64);
}

#[allow(clippy::too_many_lines)]
fn run_whir_pcs<F>(args: &Args)
where
    F: FftField + CanonicalSerialize + Codec,
{
    use whir::protocols::whir::Config;

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
    let hash_id = args.hash.hash_id();

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

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
        hash_id,
    };

    let params = Config::<F>::new(mv_params, &whir_params);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params.check_max_pow_bits(Bits::new(whir_params.pow_bits as f64)) {
        println!("WARN: more PoW bits required than what specified.");
    }

    let vector = (0..num_coeffs)
        .map(<F as Field>::BasePrimeField::from)
        .collect::<Vec<_>>();

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &[&vector]);
    let whir_commit_time = whir_commit_time.elapsed();

    let mut linear_forms: Vec<Box<dyn Evaluate<Basefield<F>>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<F>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![F::from(x as u64); num_variables]))
        .collect();

    for point in &points {
        let lf_eval = MultilinearExtension::new(point.0.clone());
        let lf_prove = MultilinearExtension::new(point.0.clone());
        evaluations.push(lf_eval.evaluate(params.embedding(), &vector));
        linear_forms.push(Box::new(lf_eval));
        prove_linear_forms.push(Box::new(lf_prove));
    }

    // Linear constraint
    for _ in 0..num_linear_constraints {
        let cv_eval = Covector {
            deferred: false,
            vector: (0..num_coeffs).map(F::from).collect(),
        };
        let cv_prove = Covector {
            deferred: false,
            vector: (0..num_coeffs).map(F::from).collect(),
        };
        evaluations.push(cv_eval.evaluate(params.embedding(), &vector));
        linear_forms.push(Box::new(cv_eval));
        prove_linear_forms.push(Box::new(cv_prove));
    }

    let whir_prove_time = Instant::now();
    params.prove(
        &mut prover_state,
        vec![Cow::Borrowed(vector.as_slice())],
        vec![Cow::Owned(witness)],
        &prove_linear_forms,
        Cow::Borrowed(evaluations.as_slice()),
    );
    let whir_prove_time = whir_prove_time.elapsed();

    let proof = prover_state.proof();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    let weight_dyn_refs = linear_forms
        .iter()
        .map(|w| w.as_ref() as &dyn LinearForm<F>)
        .collect::<Vec<_>>();

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitment = params.receive_commitment(&mut verifier_state).unwrap();
        params
            .verify(
                &mut verifier_state,
                &[&commitment],
                &weight_dyn_refs,
                &evaluations,
            )
            .unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}

#[allow(clippy::too_many_lines)]
fn run_whir_pcs_zk<F>(args: &Args)
where
    F: FftField + PrimeField + CanonicalSerialize + Codec,
{
    use whir::protocols::whir_zk::Config;

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
    let hash_id = args.hash.hash_id();

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

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
        hash_id,
    };

    let params = Config::<F>::new(mv_params, &whir_params, whir_params.folding_factor, 1);

    let ds = DomainSeparator::protocol(&params)
        .session(&format!("Example at {}:{}", file!(), line!()))
        .instance(&Empty);

    let mut prover_state = ProverState::new_std(&ds);

    println!("=========================================");
    println!("Whir (PCS + ZK) 🌪️");
    println!("Field: {:?} and hash: {:?}", args.field, args.hash);
    println!("{params}");
    if !params
        .blinded_commitment
        .check_max_pow_bits(Bits::new(whir_params.pow_bits as f64))
    {
        println!("WARN: more PoW bits required than what specified.");
    }

    let vector = (0..num_coeffs).map(F::from).collect::<Vec<_>>();

    let mut linear_forms: Vec<Box<dyn Evaluate<Identity<F>>>> = Vec::new();
    let mut prove_linear_forms: Vec<Box<dyn LinearForm<F>>> = Vec::new();
    let mut evaluations = Vec::new();

    // Evaluation constraint
    let points: Vec<_> = (0..num_evaluations)
        .map(|x| MultilinearPoint(vec![F::from(x as u64); num_variables]))
        .collect();

    for point in &points {
        let lf_eval = MultilinearExtension::new(point.0.clone());
        let lf_prove = MultilinearExtension::new(point.0.clone());
        evaluations.push(lf_eval.evaluate(params.blinded_commitment.embedding(), &vector));
        linear_forms.push(Box::new(lf_eval));
        prove_linear_forms.push(Box::new(lf_prove));
    }

    // Linear constraint
    for _ in 0..num_linear_constraints {
        let cv_eval = Covector {
            deferred: false,
            vector: (0..num_coeffs).map(F::from).collect(),
        };
        let cv_prove = Covector {
            deferred: false,
            vector: (0..num_coeffs).map(F::from).collect(),
        };
        evaluations.push(cv_eval.evaluate(params.blinded_commitment.embedding(), &vector));
        linear_forms.push(Box::new(cv_eval));
        prove_linear_forms.push(Box::new(cv_prove));
    }

    let whir_commit_time = Instant::now();
    let witness = params.commit(&mut prover_state, &[vector.as_slice()]);
    let whir_commit_time = whir_commit_time.elapsed();

    let whir_prove_time = Instant::now();
    params.prove(
        &mut prover_state,
        &[Cow::Borrowed(&vector)],
        witness,
        &prove_linear_forms,
        &evaluations,
    );
    let whir_prove_time = whir_prove_time.elapsed();

    let proof = prover_state.proof();
    println!(
        "Prover time: {whir_commit_time:.1?} + {whir_prove_time:.1?} = {:.1?}",
        whir_commit_time + whir_prove_time,
    );
    println!(
        "Proof size: {:.1} KiB",
        (proof.narg_string.len() + proof.hints.len()) as f64 / 1024.0
    );

    let weight_dyn_refs = linear_forms
        .iter()
        .map(|w| w.as_ref() as &dyn LinearForm<F>)
        .collect::<Vec<_>>();

    HASH_COUNTER.reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = params.receive_commitments(&mut verifier_state, 1).unwrap();
        params
            .verify(
                &mut verifier_state,
                &weight_dyn_refs,
                &evaluations,
                &commitment,
            )
            .unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HASH_COUNTER.get() as f64 / reps as f64) / 1000.0
    );
}
