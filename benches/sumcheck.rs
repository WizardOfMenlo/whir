use ark_ff::fields::Field;
use divan::{black_box, AllocProfiler, Bencher};
use whir::{
    algebra::{
        fields::Field64 as F,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    },
    protocols::sumcheck::SumcheckSingle,
    whir::statement::{Statement, Weights},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

const SIZES: &[u64] = &[1 << 16, 1 << 18, 1 << 20];

#[divan::bench(args = SIZES)]
fn sumcheck_first_round(bencher: Bencher, size: u64) {
    bencher
        .with_inputs(|| {
            let num_vars = size.trailing_zeros() as usize;
            let base_coeffs: Vec<_> = (0..size).map(F::from).collect();
            let eval_point = MultilinearPoint(vec![F::from(1); num_vars]);
            let combination_randomness = F::from(42);
            let folding_randomness = MultilinearPoint(vec![F::from(4999)]);

            // Reset everything on each iteration
            let coeffs = CoefficientList::new(black_box(base_coeffs));
            let value = coeffs.evaluate(&eval_point);

            let mut statement = Statement::new(num_vars);
            let weights = Weights::evaluation(eval_point);
            statement.add_constraint(weights, value);

            (
                coeffs,
                statement,
                combination_randomness,
                folding_randomness,
            )
        })
        .bench_values(
            |(coeffs, statement, combination_randomness, folding_randomness)| {
                let mut prover = SumcheckSingle::new(coeffs, &statement, F::ONE);
                let sumcheck_poly = prover.compute_sumcheck_polynomial();

                prover.compress(combination_randomness, &folding_randomness, &sumcheck_poly);
                black_box(prover);
            },
        );
}

fn main() {
    divan::main();
}
