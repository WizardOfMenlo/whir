use ark_ff::Field;
use nimue::plugins::ark::FieldIOPattern;
use nimue_pow::PoWIOPattern;

/// Trait for adding out-of-domain (OOD) queries and their responses to an IOPattern.
///
/// This trait allows extending an IOPattern with challenge-response interactions.
pub trait OODIOPattern<F: Field> {
    /// Adds `num_samples` OOD queries and their corresponding responses to the IOPattern.
    ///
    /// - If `num_samples > 0`, this appends:
    ///   - A challenge query labeled `"ood_query"` with `num_samples` elements.
    ///   - A corresponding response labeled `"ood_ans"` with `num_samples` elements.
    /// - If `num_samples == 0`, the IOPattern remains unchanged.
    #[must_use]
    fn add_ood(self, num_samples: usize) -> Self;
}

impl<F, IOPattern> OODIOPattern<F> for IOPattern
where
    F: Field,
    IOPattern: FieldIOPattern<F>,
{
    fn add_ood(self, num_samples: usize) -> Self {
        if num_samples > 0 {
            self.challenge_scalars(num_samples, "ood_query")
                .add_scalars(num_samples, "ood_ans")
        } else {
            self
        }
    }
}

/// Trait for adding a Proof-of-Work (PoW) challenge to an IOPattern.
///
/// This trait enables an IOPattern to include PoW challenges.
pub trait WhirPoWIOPattern {
    /// Adds a Proof-of-Work challenge to the IOPattern.
    ///
    /// - If `bits > 0`, this appends a PoW challenge labeled `"pow_queries"`.
    /// - If `bits == 0`, the IOPattern remains unchanged.
    #[must_use]
    fn pow(self, bits: f64) -> Self;
}

impl<IOPattern> WhirPoWIOPattern for IOPattern
where
    IOPattern: PoWIOPattern,
{
    fn pow(self, bits: f64) -> Self {
        if bits > 0. {
            self.challenge_pow("pow_queries")
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use nimue::IOPattern;

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_add_ood() {
        let iop = IOPattern::new("test_protocol");

        // Apply OOD query addition
        let updated_iop = <IOPattern as OODIOPattern<Field64>>::add_ood(iop.clone(), 3);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_bytes().to_vec()).unwrap();

        // Check if "ood_query" and "ood_ans" were correctly appended
        assert!(pattern_str.contains("ood_query"));
        assert!(pattern_str.contains("ood_ans"));

        // Test case where num_samples = 0 (should not modify anything)
        let unchanged_iop = <IOPattern as OODIOPattern<Field64>>::add_ood(iop, 0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }

    #[test]
    fn test_pow() {
        let iop = IOPattern::new("test_protocol");

        // Apply PoW challenge
        let updated_iop = iop.clone().pow(10.0);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_iop.as_bytes().to_vec()).unwrap();

        // Check if "pow_queries" was correctly added
        assert!(pattern_str.contains("pow_queries"));

        // Test case where bits = 0 (should not modify anything)
        let unchanged_iop = iop.pow(0.0);
        let unchanged_str = String::from_utf8(unchanged_iop.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }
}
