use std::ops::RangeInclusive;

pub fn grid(range: RangeInclusive<f64>, samples: usize) -> impl Iterator<Item = f64> {
    assert!(samples >= 2);
    (0..samples).map(move |i| {
        *range.start() + (*range.end() - *range.start()) * (i as f64) / ((samples - 1) as f64)
    })
}
