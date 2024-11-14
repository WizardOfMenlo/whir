use ark_ff::Field;
use rand::{
    distributions::{Distribution, Standard},
    Rng, RngCore,
};

use crate::utils::to_binary;

use self::hypercube::BinaryHypercubePoint;

pub mod coeffs;
pub mod evals;
pub mod fold;
pub mod gray_lag_poly;
pub mod hypercube;
pub mod sequential_lag_poly;
pub mod streaming_evaluation_helper;

/// Point (x_1,..., x_n) in F^n for some n. Often, the x_i are binary.
/// For the latter case, we also have BinaryHypercubePoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearPoint<F>(pub Vec<F>);

impl<F> MultilinearPoint<F>
where
    F: Field,
{
    /// returns the number of variables.
    pub fn n_variables(&self) -> usize {
        self.0.len()
    }

    // NOTE: Conversion BinaryHypercube <-> MultilinearPoint converts a
    // multilinear point (x1,x2,...,x_n) into the number with bit-pattern 0...0 x_1 x_2 ... x_n, provided all x_i are in {0,1}.
    // That means we pad zero bits in BinaryHypercube from the msb end and use big-endian for the actual conversion.

    /// Creates a MultilinearPoint from a BinaryHypercubePoint; the latter models the same thing, but is restricted to binary entries.
    pub fn from_binary_hypercube_point(point: BinaryHypercubePoint, num_variables: usize) -> Self {
        Self(
            to_binary(point.0, num_variables)
                .into_iter()
                .map(|x| if x { F::ONE } else { F::ZERO })
                .collect(),
        )
    }

    /// Converts to a BinaryHypercubePoint, provided the MultilinearPoint is actually in {0,1}^n.
    pub fn to_hypercube(&self) -> Option<BinaryHypercubePoint> {
        let mut counter = 0;
        for &coord in &self.0 {
            if coord == F::ZERO {
                counter <<= 1;
            } else if coord == F::ONE {
                counter = (counter << 1) + 1;
            } else {
                return None;
            }
        }

        Some(BinaryHypercubePoint(counter))
    }

    /// converts a univariate evaluation point into a multilinear one.
    ///
    /// Notably, consider the usual bijection
    /// {multilinear polys in n variables} <-> {univariate polys of deg < 2^n}
    /// f(x_1,...x_n)  <-> g(y) := f(y^(2^(n-1), ..., y^4, y^2, y).
    /// x_1^i_1 * ... *x_n^i_n <-> y^i, where (i_1,...,i_n) is the (big-endian) binary decomposition of i.
    ///
    /// expand_from_univariate maps the evaluation points to the multivariate domain, i.e.
    /// f(expand_from_univariate(y)) == g(y).
    /// in a way that is compatible with our endianness choices.
    pub fn expand_from_univariate(point: F, num_variables: usize) -> Self {
        let mut res = Vec::with_capacity(num_variables);
        let mut cur = point;
        for _ in 0..num_variables {
            res.push(cur);
            cur = cur * cur;
        }

        // Reverse so higher power is first
        res.reverse();

        MultilinearPoint(res)
    }
}

/// creates a random MultilinearPoint of length `num_variables` using the RNG `rng`.
impl<F> MultilinearPoint<F>
where
    Standard: Distribution<F>,
{
    pub fn rand(rng: &mut impl RngCore, num_variables: usize) -> Self {
        MultilinearPoint((0..num_variables).map(|_| rng.gen()).collect())
    }
}

/// creates a MultilinearPoint of length 1 from a single field element
impl<F> From<F> for MultilinearPoint<F> {
    fn from(value: F) -> Self {
        MultilinearPoint(vec![value])
    }
}

/// Compute eq(coords,point), where eq is the equality polynomial, where point is binary.
///
/// Recall that the equality polynomial eq(c, p) is defined as eq(c,p) == \prod_i c_i * p_i + (1-c_i)*(1-p_i).
/// Note that for fixed p, viewed as a polynomial in c, it is the interpolation polynomial associated to the evaluation point p in the evaluation set {0,1}^n.
pub fn eq_poly<F>(coords: &MultilinearPoint<F>, point: BinaryHypercubePoint) -> F
where
    F: Field,
{
    let mut point = point.0;
    let n_variables = coords.n_variables();
    assert!(point < (1 << n_variables)); // check that the lengths of coords and point match.

    let mut acc = F::ONE;

    for val in coords.0.iter().rev() {
        let b = point % 2;
        acc *= if b == 1 { *val } else { F::ONE - *val };
        point >>= 1;
    }

    acc
}

/// Compute eq(coords,point), where eq is the equality polynomial and where point is not neccessarily binary.
///
/// Recall that the equality polynomial eq(c, p) is defined as eq(c,p) == \prod_i c_i * p_i + (1-c_i)*(1-p_i).
/// Note that for fixed p, viewed as a polynomial in c, it is the interpolation polynomial associated to the evaluation point p in the evaluation set {0,1}^n.
pub fn eq_poly_outside<F>(coords: &MultilinearPoint<F>, point: &MultilinearPoint<F>) -> F
where
    F: Field,
{
    assert_eq!(coords.n_variables(), point.n_variables());

    let mut acc = F::ONE;

    for (&l, &r) in coords.0.iter().zip(&point.0) {
        acc *= l * r + (F::ONE - l) * (F::ONE - r);
    }

    acc
}

// TODO: Precompute two_inv?
// Alternatively, compute it directly without the general (and slow) .inverse() map.

/// Compute eq3(coords,point), where eq3 is the equality polynomial for {0,1,2}^n and point is interpreted as an element from {0,1,2}^n via (big Endian) ternary decomposition.
///
/// eq3(coords, point) is the unique polynomial of degree <=2 in each variable, s.t.
/// for coords, point in {0,1,2}^n, we have:
/// eq3(coords,point) = 1 if coords == point and 0 otherwise.
pub fn eq_poly3<F>(coords: &MultilinearPoint<F>, mut point: usize) -> F
where
    F: Field,
{
    let two = F::ONE + F::ONE;
    let two_inv = two.inverse().unwrap();

    let n_variables = coords.n_variables();
    assert!(point < 3usize.pow(n_variables as u32));

    let mut acc = F::ONE;

    // Note: This iterates over the ternary decomposition least-significant trit(?) first.
    // Since our convention is big endian, we reverse the order of coords to account for this.
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
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b00)), F::from(1));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b01)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b10)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b11)), F::from(0));

        let point = MultilinearPoint(vec![F::from(1), F::from(0)]);
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b00)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b01)), F::from(0));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b10)), F::from(1));
        assert_eq!(eq_poly(&point, BinaryHypercubePoint(0b11)), F::from(0));
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
        assert_eq!(eq_poly3(&point, 3), F::from(1)); // 3 corresponds to ternary (1,0)
        assert_eq!(eq_poly3(&point, 4), F::from(0));
        assert_eq!(eq_poly3(&point, 5), F::from(0));
        assert_eq!(eq_poly3(&point, 6), F::from(0));
        assert_eq!(eq_poly3(&point, 7), F::from(0));
        assert_eq!(eq_poly3(&point, 8), F::from(0));

        let point = MultilinearPoint(vec![F::from(0), F::from(2)]);

        assert_eq!(eq_poly3(&point, 0), F::from(0));
        assert_eq!(eq_poly3(&point, 1), F::from(0));
        assert_eq!(eq_poly3(&point, 2), F::from(1)); // 2 corresponds to ternary (0,2)
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
        assert_eq!(eq_poly3(&point, 8), F::from(1)); // 8 corresponds to ternary (2,2)
    }

    #[test]
    #[should_panic]
    fn test_equality_2() {
        let coords = MultilinearPoint(vec![F::from(0), F::from(0)]);

        // implicit length of BinaryHypercubePoint is (at least) 3, exceeding lenth of coords
        let _x = eq_poly(&coords, BinaryHypercubePoint(0b100));
    }

    #[test]
    fn expand_from_univariate() {
        let num_variables = 4;

        let point0 = MultilinearPoint::expand_from_univariate(F::from(0), num_variables);
        let point1 = MultilinearPoint::expand_from_univariate(F::from(1), num_variables);
        let point2 = MultilinearPoint::expand_from_univariate(F::from(2), num_variables);

        assert_eq!(point0.n_variables(), num_variables);
        assert_eq!(point1.n_variables(), num_variables);
        assert_eq!(point2.n_variables(), num_variables);

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
