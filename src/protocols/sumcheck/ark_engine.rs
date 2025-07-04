use std::array;

use ark_std::{One, Zero};
use rayon::iter::{IndexedParallelIterator as _, IntoParallelRefIterator, ParallelIterator as _};
use spongefish::codecs::arkworks_algebra::FieldDomainSeparator;
use tracing::instrument;

use crate::{
    utils::{pad_to_power_of_two, unzip_double_array, workload_size},
    FieldElement, R1CS,
};

/// Given `N` MLEs, optionally fold them and compute the sum of a map to `M` elements.
///
// TODO: Figure out a way to also half the mles on folding
pub fn sumcheck_fold_map_reduce<const N: usize, const M: usize>(
    mles: [&mut [FieldElement]; N],
    fold: Option<FieldElement>,
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0].len();
    assert!(size.is_power_of_two());
    assert!(size >= 2);
    assert!(mles.iter().all(|mle| mle.len() == size));

    if let Some(fold) = fold {
        assert!(size >= 4);
        let slices = mles.map(|mle| {
            let (p0, tail) = mle.split_at_mut(size / 4);
            let (p1, tail) = tail.split_at_mut(size / 4);
            let (p2, p3) = tail.split_at_mut(size / 4);
            [p0, p1, p2, p3]
        });
        sumcheck_fold_map_reduce_inner::<N, M>(slices, fold, map)
    } else {
        let slices = mles.map(|mle| mle.split_at(size / 2));
        sumcheck_map_reduce_inner::<N, M>(slices, map)
    }
}

/// Parallel sumcheck round without folding.
fn sumcheck_map_reduce_inner<const N: usize, const M: usize>(
    mles: [(&[FieldElement], &[FieldElement]); N],
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0].0.len();
    if size * N * 2 > workload_size::<FieldElement>() {
        // Split slices
        let pairs = mles.map(|(p0, p1)| (p0.split_at(size / 2), p1.split_at(size / 2)));
        let left = pairs.map(|((l0, _), (l1, _))| (l0, l1));
        let right = pairs.map(|((_, r0), (_, r1))| (r0, r1));

        // Parallel recurse
        let (l, r) = rayon::join(
            || sumcheck_map_reduce_inner(left, map),
            || sumcheck_map_reduce_inner(right, map),
        );

        // Combine results
        array::from_fn(|i| l[i] + r[i])
    } else {
        let mut result = [FieldElement::zero(); M];
        for i in 0..size {
            let e = mles.map(|(p0, p1)| (p0[i], p1[i]));
            let local = map(e);
            result.iter_mut().zip(local).for_each(|(r, l)| *r += l);
        }
        result
    }
}

fn sumcheck_fold_map_reduce_inner<const N: usize, const M: usize>(
    mut mles: [[&mut [FieldElement]; 4]; N],
    fold: FieldElement,
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0][0].len();
    if size * N * 4 > workload_size::<FieldElement>() {
        // Split slices
        let pairs = mles.map(|mles| mles.map(|p| p.split_at_mut(size / 2)));
        let (left, right) = unzip_double_array(pairs);

        // Parallel recurse
        let (l, r) = rayon::join(
            || sumcheck_fold_map_reduce_inner(left, fold, map),
            || sumcheck_fold_map_reduce_inner(right, fold, map),
        );

        // Combine results
        array::from_fn(|i| l[i] + r[i])
    } else {
        let mut result = [FieldElement::zero(); M];
        for i in 0..size {
            let e = array::from_fn(|j| {
                let mle = &mut mles[j];
                mle[0][i] += fold * (mle[2][i] - mle[0][i]);
                mle[1][i] += fold * (mle[3][i] - mle[1][i]);
                (mle[0][i], mle[1][i])
            });
            let local = map(e);
            result.iter_mut().zip(local).for_each(|(r, l)| *r += l);
        }
        result
    }
}

/// Unzip a [[(T,T); N]; M] into ([[T; N]; M],[[T; N]; M]) using move semantics
// TODO: Cleanup when <https://github.com/rust-lang/rust/issues/96097> lands
#[allow(unsafe_code)] // Required for `MaybeUninit`
fn unzip_double_array<T: Sized, const N: usize, const M: usize>(
    input: [[(T, T); N]; M],
) -> ([[T; N]; M], [[T; N]; M]) {
    // Create uninitialized memory for the output arrays
    let mut left: [[MaybeUninit<T>; N]; M] = [const { [const { MaybeUninit::uninit() }; N] }; M];
    let mut right: [[MaybeUninit<T>; N]; M] = [const { [const { MaybeUninit::uninit() }; N] }; M];

    // Move results to output arrays
    for (i, a) in input.into_iter().enumerate() {
        for (j, (l, r)) in a.into_iter().enumerate() {
            left[i][j] = MaybeUninit::new(l);
            right[i][j] = MaybeUninit::new(r);
        }
    }

    // Convert the arrays of MaybeUninit into fully initialized arrays
    // Safety: All the elements have been initialized above
    let left = left.map(|a| a.map(|u| unsafe { u.assume_init() }));
    let right = right.map(|a| a.map(|u| unsafe { u.assume_init() }));
    (left, right)
}
