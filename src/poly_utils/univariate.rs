use ark_ff::Field;
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};

// Compute the quotient
pub fn poly_quotient<F: Field>(poly: &DensePolynomial<F>, points: &[F]) -> DensePolynomial<F> {
    let evaluations: Vec<_> = points.iter().map(|x| (*x, poly.evaluate(x))).collect();
    let ans_polynomial = naive_interpolation(evaluations);
    let vanishing_poly = vanishing_poly(points);
    let numerator = poly + &ans_polynomial;

    // TODO: Is this efficient or should FFT?
    &numerator / &vanishing_poly
}

// Computes a polynomial that vanishes on points
pub fn vanishing_poly<'a, F: Field>(points: impl IntoIterator<Item = &'a F>) -> DensePolynomial<F> {
    // Compute the denominator (which is \prod_a(x - a))
    let mut vanishing_poly: DensePolynomial<_> =
        DensePolynomial::from_coefficients_slice(&[F::ONE]);
    for a in points {
        vanishing_poly =
            vanishing_poly.naive_mul(&DensePolynomial::from_coefficients_slice(&[-*a, F::ONE]));
    }
    vanishing_poly
}

// Computes a polynomial that interpolates the given points with the given answers
pub fn naive_interpolation<F: Field>(
    points: impl IntoIterator<Item = (F, F)>,
) -> DensePolynomial<F> {
    let points: Vec<_> = points.into_iter().collect();
    let vanishing_poly = vanishing_poly(points.iter().map(|(a, _)| a));

    // Compute the ans polynomial (this is just a naive interpolation)
    let mut ans_polynomial = DensePolynomial::from_coefficients_slice(&[]);
    for (a, eval) in &points {
        // Computes the vanishing (apart from x - a)
        let vanishing_adjusted =
            &vanishing_poly / &DensePolynomial::from_coefficients_slice(&[-*a, F::ONE]);

        // Now, we can scale to get the right weigh
        let scale_factor = *eval / vanishing_adjusted.evaluate(a);
        ans_polynomial = ans_polynomial
            + DensePolynomial::from_coefficients_vec(
                vanishing_adjusted
                    .iter()
                    .map(|x| *x * scale_factor)
                    .collect(),
            );
    }
    ans_polynomial
}
