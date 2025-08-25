use ark_ff::Field;
use spongefish::codecs::arkworks_algebra::FieldDomainSeparator;
use spongefish_pow::PoWDomainSeparator;

/// Trait for adding out-of-domain (OOD) queries and their responses to an DomainSeparator.
///
/// This trait allows extending an DomainSeparator with challenge-response interactions.
pub trait OODDomainSeparator<F: Field> {
    /// Adds `num_samples` OOD queries and their corresponding responses to the DomainSeparator.
    ///
    /// - If `num_samples > 0`, this appends:
    ///   - A challenge query labeled `"ood_query"` with `num_samples` elements.
    ///   - A corresponding response labeled `"ood_ans"` with `num_samples` elements.
    /// - If `num_samples == 0`, the DomainSeparator remains unchanged.
    #[must_use]
    fn add_ood(self, num_samples: usize, batch_size: usize) -> Self;
}

impl<F, DomainSeparator> OODDomainSeparator<F> for DomainSeparator
where
    F: Field,
    DomainSeparator: FieldDomainSeparator<F>,
{
    fn add_ood(mut self, num_samples: usize, batch_size: usize) -> Self {
        if num_samples > 0 && batch_size > 0 {
            self = self.challenge_scalars(num_samples, "ood_query");

            for i in 0..batch_size {
                self = self.add_scalars(num_samples, &format!("ood_ans_{i}"));
            }
        }
        self
    }
}

/// Trait for adding a Proof-of-Work (PoW) challenge to an DomainSeparator.
///
/// This trait enables an DomainSeparator to include PoW challenges.
pub trait WhirPoWDomainSeparator {
    /// Adds a Proof-of-Work challenge to the DomainSeparator.
    ///
    /// - If `bits > 0`, this appends a PoW challenge labeled `"pow_queries"`.
    /// - If `bits == 0`, the DomainSeparator remains unchanged.
    #[must_use]
    fn pow(self, bits: f64) -> Self;
}

impl<DomainSeparator> WhirPoWDomainSeparator for DomainSeparator
where
    DomainSeparator: PoWDomainSeparator,
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
    use spongefish::DomainSeparator;

    use super::*;
    use crate::crypto::fields::Field64;

    #[test]
    fn test_add_ood() {
        let domainsep: DomainSeparator = DomainSeparator::new("test_protocol");

        // Apply OOD query addition
        let updated_domainsep =
            <DomainSeparator as OODDomainSeparator<Field64>>::add_ood(domainsep.clone(), 3, 1);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_domainsep.as_bytes().to_vec()).unwrap();

        // Check if "ood_query" and "ood_ans" were correctly appended
        assert!(pattern_str.contains("ood_query"));
        assert!(pattern_str.contains("ood_ans"));

        // Test case where num_samples = 0 (should not modify anything)
        let unchanged_domainsep =
            <DomainSeparator as OODDomainSeparator<Field64>>::add_ood(domainsep, 0, 1);
        let unchanged_str = String::from_utf8(unchanged_domainsep.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }

    #[test]
    fn test_pow() {
        let domainsep: DomainSeparator = DomainSeparator::new("test_protocol");

        // Apply PoW challenge
        let updated_domainsep = domainsep.clone().pow(10.0);

        // Convert to a string for inspection
        let pattern_str = String::from_utf8(updated_domainsep.as_bytes().to_vec()).unwrap();

        // Check if "pow_queries" was correctly added
        assert!(pattern_str.contains("pow_queries"));

        // Test case where bits = 0 (should not modify anything)
        let unchanged_domainsep = domainsep.pow(0.0);
        let unchanged_str = String::from_utf8(unchanged_domainsep.as_bytes().to_vec()).unwrap();
        assert_eq!(unchanged_str, "test_protocol"); // Should remain the same
    }
}
