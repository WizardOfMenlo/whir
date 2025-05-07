use ark_ff::FftField;
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain, MixedRadixEvaluationDomain, Radix2EvaluationDomain,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

/// Represents an evaluation domain used in FFT-based polynomial arithmetic.
///
/// This domain is constructed over a multiplicative subgroup of a finite field, enabling
/// efficient Fast Fourier Transforms (FFTs).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct Domain<F>
where
    F: FftField,
{
    /// The domain defined over the base field, used for initial FFT operations.
    ///
    #[serde(with = "crate::ark_serde")]
    pub root_of_unity: F,
    #[serde(with = "crate::ark_serde")]
    pub root_of_unity_inv: F, // TODO maybe remove
    /// This is useful when operating in an extension field `F`, where `F::PrimeSubfield`
    /// represents the base field from which the extension was built.
    #[serde(with = "crate::ark_serde")]
    pub base_domain: Option<GeneralEvaluationDomain<F::BasePrimeField>>,
    /// The actual working domain used for FFT operations.
    #[serde(with = "crate::ark_serde")]
    pub backing_domain: GeneralEvaluationDomain<F>,
}

impl<F> Domain<F>
where
    F: FftField,
{
    /// Constructs a new evaluation domain for a polynomial of given `degree`.
    ///
    /// The domain size is computed as:
    /// ```ignore
    /// N = degree * 2 ^ log_rho_inv
    /// ```
    /// where `log_rho_inv` determines additional scaling.
    ///
    /// If the domain cannot be constructed, it returns `None`.
    pub fn new(degree: usize, log_rho_inv: usize) -> Option<Self> {
        let size = degree * (1 << log_rho_inv);
        let base_domain = GeneralEvaluationDomain::new(size)?;
        let backing_domain = Self::to_extension_domain(&base_domain);

        // TODO check what is base_domain and extension domain.
        let root_of_unity: F = match backing_domain {
            GeneralEvaluationDomain::Radix2(r2) => r2.group_gen,
            GeneralEvaluationDomain::MixedRadix(mr) => mr.group_gen,
        };

        let root_of_unity_inv = match backing_domain {
            GeneralEvaluationDomain::Radix2(r2) => r2.group_gen_inv,
            GeneralEvaluationDomain::MixedRadix(mr) => mr.group_gen_inv,
        };

        Some(Self {
            root_of_unity,
            root_of_unity_inv,
            backing_domain,
            base_domain: Some(base_domain),
        })
    }

    /// Returns the domain size after `folding_factor` applications of folding.
    ///
    /// Folding reduces the domain size by a factor of `2^folding_factor`, ensuring that
    /// `size` remains divisible by `2^folding_factor`. The resulting size is:
    /// ```ignore
    /// folded_size = size / 2 ^ folding_factor
    /// ```
    pub fn folded_size(&self, folding_factor: usize) -> usize {
        assert!(self.backing_domain.size() % (1 << folding_factor) == 0);
        self.backing_domain.size() >> folding_factor
    }

    /// Returns the total size of the domain.
    pub fn size(&self) -> usize {
        self.backing_domain.size()
    }

    /// Scales the domain generator by a given power, reducing its size.
    ///
    /// Scaling transforms the domain `<w>` into `<w^power>`, where:
    /// ```ignore
    /// new_size = size / power
    /// ```
    /// The base domain is set to `None` since scaling only affects the extended field.
    #[must_use]
    pub fn scale(&self, power: usize) -> Self {
        Self {
            root_of_unity: self.root_of_unity,
            root_of_unity_inv: self.root_of_unity_inv,
            backing_domain: self.scale_generator_by(power),
            base_domain: None, // Set to zero because we only care for the initial
        }
    }

    #[must_use]
    pub fn scale_with_offset(&self, power: usize) -> Self {
        Self {
            root_of_unity: self.root_of_unity,
            root_of_unity_inv: self.root_of_unity_inv,
            base_domain: None, // `base_domain` is only used for the initial commit, later on we only commit using the backing domain.
            backing_domain: self.scale_generator_with_offset(power),
        }
    }

    /// Converts a base field evaluation domain into an extended field domain.
    ///
    /// Maps elements from `F::BasePrimeField` to `F`, preserving the subgroup structure.
    fn to_extension_domain(
        domain: &GeneralEvaluationDomain<F::BasePrimeField>,
    ) -> GeneralEvaluationDomain<F> {
        let group_gen = F::from_base_prime_field(domain.group_gen());
        let group_gen_inv = F::from_base_prime_field(domain.group_gen_inv());
        let size = domain.size() as u64;
        let log_size_of_group = domain.log_size_of_group() as u32;
        let size_as_field_element = F::from_base_prime_field(domain.size_as_field_element());
        let size_inv = F::from_base_prime_field(domain.size_inv());
        let offset = F::from_base_prime_field(domain.coset_offset());
        let offset_inv = F::from_base_prime_field(domain.coset_offset_inv());
        let offset_pow_size = F::from_base_prime_field(domain.coset_offset_pow_size());
        match domain {
            GeneralEvaluationDomain::Radix2(_) => {
                GeneralEvaluationDomain::Radix2(Radix2EvaluationDomain {
                    size,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv,
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size,
                })
            }
            GeneralEvaluationDomain::MixedRadix(_) => {
                GeneralEvaluationDomain::MixedRadix(MixedRadixEvaluationDomain {
                    size,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv,
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size,
                })
            }
        }
    }

    /// Scales the domain generator by a given power.
    ///
    /// Given a domain `<w>`, this computes `<w^power>`, reducing the size:
    /// ```ignore
    /// new_size = size / power
    /// ```
    /// It ensures `size % power == 0` for a valid transformation.
    fn scale_generator_by(&self, power: usize) -> GeneralEvaluationDomain<F> {
        let starting_size = self.size();
        assert_eq!(starting_size % power, 0);
        let new_size = starting_size / power;
        let log_size_of_group = new_size.trailing_zeros();
        let size_as_field_element = F::from(new_size as u64);

        match self.backing_domain {
            GeneralEvaluationDomain::Radix2(r2) => {
                let group_gen = r2.group_gen.pow([power as u64]);
                let group_gen_inv = group_gen.inverse().unwrap();

                let offset = r2.offset.pow([power as u64]);
                let offset_inv = r2.offset_inv.pow([power as u64]);
                let offset_pow_size = offset.pow([new_size as u64]);

                GeneralEvaluationDomain::Radix2(Radix2EvaluationDomain {
                    size: new_size as u64,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv: size_as_field_element.inverse().unwrap(),
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size,
                })
            }
            GeneralEvaluationDomain::MixedRadix(mr) => {
                let group_gen = mr.group_gen.pow([power as u64]);
                let group_gen_inv = mr.group_gen_inv.pow([power as u64]);

                let offset = mr.offset.pow([power as u64]);
                let offset_inv = mr.offset_inv.pow([power as u64]);
                let offset_pow_size = offset.pow([new_size as u64]);

                GeneralEvaluationDomain::MixedRadix(MixedRadixEvaluationDomain {
                    size: new_size as u64,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv: size_as_field_element.inverse().unwrap(),
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size,
                })
            }
        }
    }

    fn scale_generator_with_offset(&self, power: usize) -> GeneralEvaluationDomain<F> {
        let starting_size = self.size();
        assert_eq!(starting_size % power, 0);
        let new_size = starting_size / power;
        let log_size_of_group = new_size.trailing_zeros();
        let size_as_field_element = F::from(new_size as u64);
        match self.backing_domain {
            GeneralEvaluationDomain::Radix2(r2) => {
                let group_gen = r2.group_gen.pow([power as u64]);
                let group_gen_inv = r2.group_gen_inv.pow([power as u64]);

                let offset = r2.offset.pow([power as u64]) * self.root_of_unity;
                let offset_inv = r2.offset_inv.pow([power as u64]) * self.root_of_unity_inv;

                GeneralEvaluationDomain::Radix2(Radix2EvaluationDomain {
                    size: new_size as u64,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv: size_as_field_element.inverse().unwrap(),
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size: offset.pow([new_size as u64]),
                })
            }
            GeneralEvaluationDomain::MixedRadix(mr) => {
                let group_gen = mr.group_gen.pow([power as u64]);
                let group_gen_inv = mr.group_gen_inv.pow([power as u64]);

                let offset = mr.offset.pow([power as u64]) * self.root_of_unity;
                let offset_inv = mr.offset_inv.pow([power as u64]) * self.root_of_unity_inv;

                GeneralEvaluationDomain::MixedRadix(MixedRadixEvaluationDomain {
                    size: new_size as u64,
                    log_size_of_group,
                    size_as_field_element,
                    size_inv: size_as_field_element.inverse().unwrap(),
                    group_gen,
                    group_gen_inv,
                    offset,
                    offset_inv,
                    offset_pow_size: offset.pow([new_size as u64]),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::crypto::fields::{Field64, Field64_2};

    type F = Field64;
    type EF2 = Field64_2;

    #[test]
    fn test_domain_creation_valid() {
        // We choose degree = 8 and log_rho_inv = 0, so the expected size is:
        // size = degree * 2^log_rho_inv = 8 * 2^0 = 8
        let domain = Domain::<F>::new(8, 0).unwrap();
        assert_eq!(domain.base_domain.as_ref().unwrap().size(), 8);
        assert_eq!(domain.backing_domain.size(), 8);
    }

    #[test]
    fn test_domain_creation_invalid() {
        // We try to create a domain with size larger than F's TWO_ADICITY limit.
        // F::TWO_ADICITY = 27, so we pick a size beyond 2^27.
        let invalid_size = 1 << (F::TWO_ADICITY + 1);
        assert!(Domain::<F>::new(invalid_size, 0).is_none());
    }

    #[test]
    fn test_base_domain_conversion() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let base_domain = domain.base_domain.as_ref().unwrap();

        // Check the domain size
        assert_eq!(base_domain.size(), 16);

        // The generator should satisfy g^(size) = 1
        let group_gen = base_domain.group_gen();
        assert_eq!(group_gen.pow([16]), F::ONE);
    }

    #[test]
    fn test_backing_domain_conversion() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let base_domain = domain.base_domain.as_ref().unwrap();
        let backing_domain = &domain.backing_domain;

        // Ensure the backing domain is derived correctly from the base domain
        assert_eq!(backing_domain.size(), base_domain.size());
        assert_eq!(
            backing_domain.group_gen(),
            F::from_base_prime_field(base_domain.group_gen())
        );

        // Verify inverse generator relation: g * g⁻¹ = 1
        let g = backing_domain.group_gen();
        let g_inv = backing_domain.group_gen_inv();
        assert_eq!(g * g_inv, F::ONE);
    }

    #[test]
    fn test_coset_offsets() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // Coset offset should be 1 in default case
        assert_eq!(backing_domain.coset_offset(), F::ONE);
        assert_eq!(backing_domain.coset_offset_inv(), F::ONE);

        // Offset raised to size should be 1: offset^size = 1
        let offset = backing_domain.coset_offset();
        assert_eq!(offset.pow([backing_domain.size() as u64]), F::ONE);
    }

    #[test]
    fn test_size_as_field_element() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // Check if size_as_field_element correctly converts the size to field representation
        assert_eq!(backing_domain.size_as_field_element(), F::from(16));
    }

    #[test]
    fn test_size_inv() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let backing_domain = &domain.backing_domain;

        // size_inv should be the multiplicative inverse of size in the field
        let size_inv = backing_domain.size_inv();
        assert_eq!(size_inv * F::from(16), F::ONE);
    }

    #[test]
    fn test_folded_size_valid() {
        let domain = Domain::<F>::new(16, 0).unwrap();

        // Folding factor = 2 → New size = size / (2^2) = 16 / 4 = 4
        assert_eq!(domain.folded_size(2), 4);
    }

    #[test]
    #[should_panic]
    fn test_folded_size_invalid() {
        let domain = Domain::<F>::new(10, 0).unwrap();
        // This should panic since 16 is not divisible by 5^2
        domain.folded_size(5);
    }

    #[test]
    fn test_scaling_preserves_structure() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale(2);

        // The scaled domain should have the size divided by 2.
        assert_eq!(scaled_domain.size(), 8);

        // The generator of the scaled domain should be `g^2`.
        let expected_group_gen = domain.backing_domain.group_gen().pow([2]);
        assert_eq!(scaled_domain.backing_domain.group_gen(), expected_group_gen);

        // The inverse generator should be `g^-2`.
        let expected_group_gen_inv = expected_group_gen.inverse().unwrap();
        assert_eq!(
            scaled_domain.backing_domain.group_gen_inv(),
            expected_group_gen_inv
        );
    }

    #[test]
    fn test_scale_generator_by_valid() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New size = size / power = 16 / 2 = 8
        assert_eq!(scaled_domain.size(), 8);

        // New generator should be g^2
        let expected_group_gen = domain.backing_domain.group_gen().pow([2]);
        assert_eq!(scaled_domain.group_gen(), expected_group_gen);

        // New inverse generator should be (g^2)^-1
        let expected_group_gen_inv = expected_group_gen.inverse().unwrap();
        assert_eq!(scaled_domain.group_gen_inv(), expected_group_gen_inv);
    }

    #[test]
    #[should_panic]
    fn test_scale_generator_by_invalid() {
        let domain = Domain::<F>::new(10, 0).unwrap();
        // This should panic since size is not divisible by 3
        domain.scale_generator_by(3);
    }

    #[test]
    fn test_offsets_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New domain size should be 16 / 2 = 8
        assert_eq!(scaled_domain.size(), 8);

        // The offset should be raised to the power of `power`
        let expected_offset = domain.backing_domain.coset_offset().pow([2]);
        assert_eq!(scaled_domain.coset_offset(), expected_offset);

        // The inverse offset should be raised to the power of `power`
        let expected_offset_inv = domain.backing_domain.coset_offset_inv().pow([2]);
        assert_eq!(scaled_domain.coset_offset_inv(), expected_offset_inv);

        // The offset_pow_size should be offset^(new_size)
        let expected_offset_pow_size = expected_offset.pow([8]);
        assert_eq!(
            scaled_domain.coset_offset_pow_size(),
            expected_offset_pow_size
        );
    }

    #[test]
    fn test_size_as_field_element_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // New domain size should be 16 / 2 = 8
        let expected_size_as_field_element = F::from(8);

        // Check if size_as_field_element correctly represents the scaled size in the field
        assert_eq!(
            scaled_domain.size_as_field_element(),
            expected_size_as_field_element
        );
    }

    #[test]
    fn test_log_size_of_group_after_scaling() {
        let domain = Domain::<F>::new(16, 0).unwrap();
        let scaled_domain = domain.scale_generator_by(2);

        // The original size is 16, so log_size_of_group should be log2(16) = 4.
        assert_eq!(domain.backing_domain.log_size_of_group(), 4);

        // After scaling by 2, the new size is 16 / 2 = 8, so log_size_of_group should be log2(8) =
        // 3.
        assert_eq!(scaled_domain.log_size_of_group(), 3);
    }

    #[test]
    fn test_domain_creation_ef2_valid() {
        // Creates a domain over EF4, but constructs it by lifting from a base domain in F.
        // This ensures that to_extension_domain works properly.
        let domain = Domain::<EF2>::new(8, 0).unwrap();

        // Base and backing domains should both be of size 8.
        assert_eq!(domain.base_domain.as_ref().unwrap().size(), 8);
        assert_eq!(domain.backing_domain.size(), 8);
    }

    #[test]
    fn test_domain_creation_ef2_invalid() {
        // If the base domain cannot be created due to TWO_ADICITY limits, the constructor should fail.
        let invalid_size = 1 << (F::TWO_ADICITY + 1);
        assert!(Domain::<EF2>::new(invalid_size, 0).is_none());
    }

    #[test]
    fn test_to_extension_domain_matches_structure() {
        // We want to verify that all the components of a GeneralEvaluationDomain<F>
        // are correctly lifted into the extension field EF4.
        let base = GeneralEvaluationDomain::<F>::new(16).unwrap();
        let ext = Domain::<EF2>::to_extension_domain(&base);

        // The extension domain should have identical size and group generator structure,
        // just lifted to EF4 via from_prime_subfield.
        assert_eq!(ext.size(), base.size());
        assert_eq!(
            ext.group_gen(),
            EF2::from_base_prime_field(base.group_gen())
        );
        assert_eq!(
            ext.coset_offset(),
            EF2::from_base_prime_field(base.coset_offset())
        );
    }

    #[test]
    fn test_scaling_preserves_structure_ef2() {
        // Scaling should preserve the multiplicative structure of the group in EF4.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        let scaled_domain = domain.scale(2); // Halve the size, square the generator

        // Size should be halved
        assert_eq!(scaled_domain.size(), 8);

        // Generator should be squared
        let expected_gen = domain.backing_domain.group_gen().pow([2_u64]);
        assert_eq!(scaled_domain.backing_domain.group_gen(), expected_gen);

        // Inverse generator should be inverse of g²
        assert_eq!(
            scaled_domain.backing_domain.group_gen_inv(),
            expected_gen.inverse().unwrap()
        );
    }

    #[test]
    fn test_folded_size_ef2_valid() {
        // Verify that the folded size logic works over EF4.
        // 16 elements → fold by factor 4 → expect size 4
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        assert_eq!(domain.folded_size(2), 4);
    }

    #[test]
    fn test_scale_generator_by_valid_ef2() {
        // We directly test the internal generator scaling function with EF4.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        // Should result in domain of size 8
        assert_eq!(scaled.size(), 8);

        // The generator must be g² and the inverse g⁻²
        let expected_gen = domain.backing_domain.group_gen().pow([2_u64]);
        assert_eq!(scaled.group_gen(), expected_gen);
        assert_eq!(scaled.group_gen_inv(), expected_gen.inverse().unwrap());
    }

    #[test]
    #[should_panic]
    fn test_scale_generator_by_invalid_ef2() {
        // If the domain size is not divisible by the scaling factor,
        // this should panic due to assertion.
        let domain = Domain::<EF2>::new(10, 0).unwrap();
        domain.scale_generator_by(3); // 10 not divisible by 3
    }

    #[test]
    fn test_offsets_after_scaling_ef2() {
        // This test ensures that the coset offset is scaled correctly along with the generator.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        // offset' = offset^2, offset_inv' = offset_inv^2
        let expected_offset = domain.backing_domain.coset_offset().pow([2_u64]);
        let expected_inv = domain.backing_domain.coset_offset_inv().pow([2_u64]);

        // offset_pow_size' = offset'^new_size
        let expected_pow = expected_offset.pow([8_u64]);

        assert_eq!(scaled.coset_offset(), expected_offset);
        assert_eq!(scaled.coset_offset_inv(), expected_inv);
        assert_eq!(scaled.coset_offset_pow_size(), expected_pow);
    }

    #[test]
    fn test_size_as_field_element_ef2() {
        // The domain size as an EF2 field element should match the u64 conversion.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        assert_eq!(domain.backing_domain.size_as_field_element(), EF2::from(16));
    }

    #[test]
    fn test_size_inv_ef2() {
        // size_inv should be the true multiplicative inverse of size in EF2.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        let inv = domain.backing_domain.size_inv();
        assert_eq!(inv * EF2::from(16), EF2::ONE);
    }

    #[test]
    fn test_log_size_of_group_after_scaling_ef2() {
        // The log of the domain size (in base 2) should decrease by 1 when scaling by 2.
        let domain = Domain::<EF2>::new(16, 0).unwrap();
        let scaled = domain.scale_generator_by(2);

        assert_eq!(domain.backing_domain.log_size_of_group(), 4); // log2(16)
        assert_eq!(scaled.log_size_of_group(), 3); // log2(8)
    }
}
