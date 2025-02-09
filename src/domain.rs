use ark_ff::FftField;
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain, MixedRadixEvaluationDomain, Radix2EvaluationDomain,
};

#[derive(Debug, Clone)]
pub struct Domain<F>
where
    F: FftField,
{
    pub root_of_unity: F,
    pub root_of_unity_inv: F, // TODO maybe remove
    pub base_domain: Option<GeneralEvaluationDomain<F::BasePrimeField>>, // The domain (in the base
    // field) for the initial FFT
    pub backing_domain: GeneralEvaluationDomain<F>,
}

impl<F> Domain<F>
where
    F: FftField,
{
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

    pub fn folded_size(&self, folding_factor: usize) -> usize {
        self.backing_domain.size() / (1 << folding_factor)
    }

    pub fn size(&self) -> usize {
        self.backing_domain.size()
    }

    pub fn scale(&self, power: usize) -> Self {
        Self {
            root_of_unity: self.root_of_unity,
            root_of_unity_inv: self.root_of_unity_inv,
            backing_domain: self.scale_generator_by(power),
            base_domain: None, // Set to zero because we only care for the initial
        }
    }

    pub fn scale_with_offset(&self, power: usize) -> Self {
        Self {
            root_of_unity: self.root_of_unity,
            root_of_unity_inv: self.root_of_unity_inv,
            base_domain: None, // TODO why?
            backing_domain: self.scale_generator_with_offset(power),
        }
    }

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

    // Takes the underlying backing_domain = <w>, and computes the new domain
    // <w^power> (note this will have size |L| / power)
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
