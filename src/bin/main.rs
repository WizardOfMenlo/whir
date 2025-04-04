use std::time::Instant;

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
};
use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use clap::Parser;
use spongefish::{DomainSeparator, ProverState, VerifierState};
use spongefish_pow::blake3::Blake3PoW;
use whir::{
    cmdline_utils::{AvailableFields, AvailableMerkle, WhirType},
    crypto::{
        fields,
        merkle_tree::{
            self,
            blake3::{Blake3Compress, Blake3LeafHash, Blake3MerkleTreeParams},
            keccak::{KeccakCompress, KeccakLeafHash, KeccakMerkleTreeParams},
            HashCounter,
        },
    },
    parameters::{
        default_max_pow, FoldType, FoldingFactor, MultivariateParameters, ProtocolParameters,
        SoundnessType,
    },
    poly_utils::{coeffs::CoefficientList, evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::CommitmentReader,
        domainsep::DigestDomainSeparator,
        statement::{Statement, StatementVerifier, Weights},
        utils::{DigestToUnitDeserialize, DigestToUnitSerialize},
    },
};

use crate::merkle_tree::parameters::default_config;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 't', long = "type", default_value = "PCS")]
    protocol_type: WhirType,

    #[arg(short = 'l', long, default_value = "100")]
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

    #[arg(long = "sec", default_value = "ConjectureList")]
    soundness_type: SoundnessType,

    #[arg(long = "fold_type", default_value = "ProverHelps")]
    fold_optimisation: FoldType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks2")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    merkle_tree: AvailableMerkle,
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

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks1, AvailableMerkle::Keccak256) => {
            use fields::Field64 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Blake3) => {
            use fields::Field64_2 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Keccak256) => {
            use fields::Field64_2 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Blake3) => {
            use fields::Field64_3 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Keccak256) => {
            use fields::Field64_3 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Blake3) => {
            use fields::Field128 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Keccak256) => {
            use fields::Field128 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Blake3) => {
            use fields::Field192 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Keccak256) => {
            use fields::Field192 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Blake3) => {
            use fields::Field256 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, Blake3LeafHash<F>, Blake3Compress>(&mut rng);
            run_whir::<F, Blake3MerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Keccak256) => {
            use fields::Field256 as F;

            let (leaf_hash_params, two_to_one_params) =
                default_config::<F, KeccakLeafHash<F>, KeccakCompress>(&mut rng);
            run_whir::<F, KeccakMerkleTreeParams<F>>(&args, leaf_hash_params, two_to_one_params);
        }
    }
}

fn run_whir<F, MerkleConfig>(
    args: &Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    DomainSeparator: DigestDomainSeparator<MerkleConfig>,
    ProverState: DigestToUnitSerialize<MerkleConfig>,
    for<'a> VerifierState<'a>: DigestToUnitDeserialize<MerkleConfig>,
{
    match args.protocol_type {
        WhirType::PCS => {
            run_whir_pcs::<F, MerkleConfig>(args, leaf_hash_params, two_to_one_params);
        }
        WhirType::LDT => {
            run_whir_as_ldt::<F, MerkleConfig>(args, leaf_hash_params, two_to_one_params);
        }
    }
}

fn run_whir_as_ldt<F, MerkleConfig>(
    args: &Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    DomainSeparator: DigestDomainSeparator<MerkleConfig>,
    ProverState: DigestToUnitSerialize<MerkleConfig>,
    for<'a> VerifierState<'a>: DigestToUnitDeserialize<MerkleConfig>,
{
    use whir::whir::{
        committer::CommitmentWriter, domainsep::WhirDomainSeparator, parameters::WhirConfig,
        prover::Prover, verifier::Verifier,
    };

    // Runs as a LDT
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let fold_optimisation = args.fold_optimisation;
    let soundness_type = args.soundness_type;

    if args.num_evaluations > 1 {
        println!("Warning: running as LDT but a number of evaluations to be proven was specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters::<MerkleConfig, PowStrategy> {
        initial_statement: false,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        leaf_hash_params,
        two_to_one_params,
        soundness_type,
        fold_optimisation,
        _pow_parameters: Default::default(),
        starting_log_inv_rate: starting_rate,
    };

    let params = WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params);

    let domainsep = DomainSeparator::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params);

    let mut prover_state = domainsep.to_prover_state();

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
    let witness = committer.commit(&mut prover_state, polynomial).unwrap();

    let prover = Prover(params.clone());

    let statement = Statement::new(num_variables);
    let statement_verifier = StatementVerifier::from_statement(&statement);
    let proof = prover.prove(&mut prover_state, statement, witness).unwrap();

    dbg!(whir_prover_time.elapsed());

    // Serialize proof
    let narg_string = prover_state.narg_string().to_vec();
    let mut proof_bytes = vec![];
    proof.serialize_compressed(&mut proof_bytes).unwrap();

    let proof_size = narg_string.len() + proof_bytes.len();
    dbg!(proof_size);

    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = domainsep.to_verifier_state(&narg_string);
        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();
        verifier
            .verify(
                &mut verifier_state,
                &parsed_commitment,
                &statement_verifier,
                &proof,
            )
            .unwrap();
    }
    dbg!(whir_verifier_time.elapsed() / reps as u32);
    dbg!(HashCounter::get() as f64 / reps as f64);
}

#[allow(clippy::too_many_lines)]
fn run_whir_pcs<F, MerkleConfig>(
    args: &Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    DomainSeparator: DigestDomainSeparator<MerkleConfig>,
    ProverState: DigestToUnitSerialize<MerkleConfig>,
    for<'a> VerifierState<'a>: DigestToUnitDeserialize<MerkleConfig>,
{
    use whir::whir::{
        committer::CommitmentWriter, domainsep::WhirDomainSeparator, parameters::WhirConfig,
        prover::Prover, statement::Statement, verifier::Verifier, whir_proof_size,
    };

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let first_round_folding_factor = args.first_round_folding_factor;
    let folding_factor = args.folding_factor;
    let fold_optimisation = args.fold_optimisation;
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;
    let num_linear_constraints = args.num_linear_constraints;

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = ProtocolParameters::<MerkleConfig, PowStrategy> {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor: FoldingFactor::ConstantFromSecondRound(
            first_round_folding_factor,
            folding_factor,
        ),
        leaf_hash_params,
        two_to_one_params,
        soundness_type,
        fold_optimisation,
        _pow_parameters: Default::default(),
        starting_log_inv_rate: starting_rate,
    };

    let params = WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params);

    let domainsep = DomainSeparator::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params);

    let mut prover_state = domainsep.to_prover_state();

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
    let witness = committer
        .commit(&mut prover_state, polynomial.clone())
        .unwrap();

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

    let prover = Prover(params.clone());

    let proof = prover
        .prove(&mut prover_state, statement.clone(), witness)
        .unwrap();

    println!("Prover time: {:.1?}", whir_prover_time.elapsed());
    println!(
        "Proof size: {:.1} KiB",
        whir_proof_size(prover_state.narg_string(), &proof) as f64 / 1024.0
    );

    let statement_verifier = StatementVerifier::from_statement(&statement);
    // Just not to count that initial inversion (which could be precomputed)
    let commitment_reader = CommitmentReader::new(&params);
    let verifier = Verifier::new(&params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut verifier_state = domainsep.to_verifier_state(prover_state.narg_string());
        let parsed_commitment = commitment_reader
            .parse_commitment(&mut verifier_state)
            .unwrap();
        verifier
            .verify(
                &mut verifier_state,
                &parsed_commitment,
                &statement_verifier,
                &proof,
            )
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
