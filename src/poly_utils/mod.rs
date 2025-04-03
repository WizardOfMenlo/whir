use crate::poly_utils::multilinear::MultilinearPoint;
use crate::utils::to_binary;
use ark_ff::Field;
use rand::{
    distributions::{Distribution, Standard},
    Rng, RngCore,
};

use self::hypercube::BinaryHypercubePoint;

pub mod coeffs;
pub mod dense;
pub mod evals;
pub mod fold;
pub mod hypercube;
pub mod univariate;

// #[derive(Debug, Clone, PartialEq, Eq)]
// pub struct MultilinearPoint<F>(pub Vec<F>);

pub fn eq_poly<F>(coords: &MultilinearPoint<F>, point: BinaryHypercubePoint) -> F
where
    F: Field,
{
    let mut point = point.0;
    let n_variables = coords.num_variables();
    assert!(point < (1 << n_variables));

    let mut acc = F::ONE;

    for val in coords.0.iter().rev() {
        let b = point % 2;
        acc *= if b == 1 { *val } else { F::ONE - *val };
        point >>= 1;
    }

    acc
}

pub fn eq_poly_outside<F>(coords: &MultilinearPoint<F>, point: &MultilinearPoint<F>) -> F
where
    F: Field,
{
    assert_eq!(coords.num_variables(), point.num_variables());

    let mut acc = F::ONE;

    for (&l, &r) in coords.0.iter().zip(&point.0) {
        acc *= l * r + (F::ONE - l) * (F::ONE - r);
    }

    acc
}

pub fn eq_poly3<F>(coords: &MultilinearPoint<F>, mut point: usize) -> F
where
    F: Field,
{
    let two = F::ONE + F::ONE;
    let two_inv = two.inverse().unwrap();

    let n_variables = coords.num_variables();
    assert!(point < 3usize.pow(n_variables as u32));

    let mut acc = F::ONE;

    for &val in coords.0.iter().rev() {
        let b = point % 3;
        acc *= match b {
            0 => (val - F::ONE) * (val - two) * two_inv,
            1 => val * (val - two) * (-F::ONE),
            2 => val * (val - F::ONE) * two_inv,
            _ => unreachable!(),
        };
        point /= 3;
    }

    acc
}

#[cfg(test)]
mod tests {
    use crate::poly_utils::eq_poly3;
    use crate::poly_utils::hypercube::BinaryHypercube;
    use crate::{crypto::fields::Field64, poly_utils::eq_poly};

    use super::coeffs::CoefficientList;
    use super::BinaryHypercubePoint;
    use super::MultilinearPoint;

    type F = Field64;

    #[test]
    fn test_equality() {
        let point = MultilinearPoint(vec![F::from(0), F::from(0)]);
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0)), F::from(1));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(1)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(2)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(3)), F::from(0));

        let point = MultilinearPoint(vec![F::from(1), F::from(0)]);
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(1)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(2)), F::from(1));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(3)), F::from(0));
    }

    #[test]
    fn test_equality_again() {
        let poly = CoefficientList::new(vec![F::from(35), F::from(97), F::from(10), F::from(32)]);
        let point = MultilinearPoint(vec![F::from(42), F::from(36)]);
        let eval = poly.evaluate(&point);

        assert_eq!(
            eval,
            BinaryHypercube::new(2)
                .map(
                    |i| poly.evaluate(&MultilinearPoint::from_binary_hypercube_point(i, 2))
                        * eq_poly(&point, i)
                )
                .sum()
        );
    }

    #[test]
    fn test_equality3() {
        let point = MultilinearPoint(vec![F::from(0), F::from(0)]);

        assert_eq!(eq_poly3(&point, 0), F::from(1));
        assert_eq!(eq_poly3(&point, 1), F::from(0));
        assert_eq!(eq_poly3(&point, 2), F::from(0));
        assert_eq!(eq_poly3(&point, 3), F::from(0));
        assert_eq!(eq_poly3(&point, 4), F::from(0));
        assert_eq!(eq_poly3(&point, 5), F::from(0));
        assert_eq!(eq_poly3(&point, 6), F::from(0));
        assert_eq!(eq_poly3(&point, 7), F::from(0));
        assert_eq!(eq_poly3(&point, 8), F::from(0));

        let point = MultilinearPoint(vec![F::from(1), F::from(0)]);

        assert_eq!(eq_poly3(&point, 0), F::from(0));
        assert_eq!(eq_poly3(&point, 1), F::from(0));
        assert_eq!(eq_poly3(&point, 2), F::from(0));
        assert_eq!(eq_poly3(&point, 3), F::from(1));
        assert_eq!(eq_poly3(&point, 4), F::from(0));
        assert_eq!(eq_poly3(&point, 5), F::from(0));
        assert_eq!(eq_poly3(&point, 6), F::from(0));
        assert_eq!(eq_poly3(&point, 7), F::from(0));
        assert_eq!(eq_poly3(&point, 8), F::from(0));

        let point = MultilinearPoint(vec![F::from(0), F::from(2)]);

        assert_eq!(eq_poly3(&point, 0), F::from(0));
        assert_eq!(eq_poly3(&point, 1), F::from(0));
        assert_eq!(eq_poly3(&point, 2), F::from(1));
        assert_eq!(eq_poly3(&point, 3), F::from(0));
        assert_eq!(eq_poly3(&point, 4), F::from(0));
        assert_eq!(eq_poly3(&point, 5), F::from(0));
        assert_eq!(eq_poly3(&point, 6), F::from(0));
        assert_eq!(eq_poly3(&point, 7), F::from(0));
        assert_eq!(eq_poly3(&point, 8), F::from(0));

        let point = MultilinearPoint(vec![F::from(2), F::from(2)]);

        assert_eq!(eq_poly3(&point, 0), F::from(0));
        assert_eq!(eq_poly3(&point, 1), F::from(0));
        assert_eq!(eq_poly3(&point, 2), F::from(0));
        assert_eq!(eq_poly3(&point, 3), F::from(0));
        assert_eq!(eq_poly3(&point, 4), F::from(0));
        assert_eq!(eq_poly3(&point, 5), F::from(0));
        assert_eq!(eq_poly3(&point, 6), F::from(0));
        assert_eq!(eq_poly3(&point, 7), F::from(0));
        assert_eq!(eq_poly3(&point, 8), F::from(1));
    }

    #[test]
    #[should_panic]
    fn test_equality_2() {
        let point = MultilinearPoint(vec![F::from(0), F::from(0)]);

        let _x = eq_poly(&point, BinaryHypercubePoint(4));
    }

    #[test]
    fn expand_from_univariate() {
        let num_variables = 4;

        let point0 = MultilinearPoint::expand_from_univariate(F::from(0), num_variables);
        let point1 = MultilinearPoint::expand_from_univariate(F::from(1), num_variables);
        let point2 = MultilinearPoint::expand_from_univariate(F::from(2), num_variables);

        assert_eq!(point0.num_variables(), num_variables);
        assert_eq!(point1.num_variables(), num_variables);
        assert_eq!(point2.num_variables(), num_variables);

        assert_eq!(
            MultilinearPoint::from_binary_hypercube_point(BinaryHypercubePoint(0), num_variables),
            point0
        );

        assert_eq!(
            MultilinearPoint::from_binary_hypercube_point(
                BinaryHypercubePoint((1 << num_variables) - 1),
                num_variables
            ),
            point1
        );

        assert_eq!(
            MultilinearPoint(vec![F::from(256), F::from(16), F::from(4), F::from(2)]),
            point2
        );
    }

    #[test]
    fn from_hypercube_and_back() {
        let hypercube_point = BinaryHypercubePoint(24);
        assert_eq!(
            Some(hypercube_point),
            MultilinearPoint::<F>::from_binary_hypercube_point(hypercube_point, 5).to_hypercube()
        );
    }
}
pub mod lagrange_iterator;
pub mod multilinear;
