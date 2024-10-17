use ark_ff::{
    Field, Fp128, Fp192, Fp2, Fp256, Fp2Config, Fp3, Fp3Config, Fp64, MontBackend, MontConfig,
    MontFp, PrimeField,
};
use serde::Serialize;

pub trait FieldWithSize {
    fn field_size_in_bits() -> usize;
}

impl<F> FieldWithSize for F
where
    F: Field,
{
    fn field_size_in_bits() -> usize {
        F::BasePrimeField::MODULUS_BIT_SIZE as usize * F::extension_degree() as usize
    }
}

#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "5"]
pub struct BN254Config;
pub type Field256 = Fp256<MontBackend<BN254Config, 4>>;

#[derive(MontConfig)]
#[modulus = "3801539170989320091464968600173246866371124347557388484609"]
#[generator = "3"]
pub struct FConfig192;
pub type Field192 = Fp192<MontBackend<FConfig192, 3>>;

#[derive(MontConfig)]
#[modulus = "340282366920938463463374557953744961537"]
#[generator = "3"]
pub struct FrConfig128;
pub type Field128 = Fp128<MontBackend<FrConfig128, 2>>;

#[derive(MontConfig, Serialize)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "5"]
pub struct FqConfig;
pub type FieldBn256 = Fp256<MontBackend<FqConfig, 4>>;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FConfig64;
pub type Field64 = Fp64<MontBackend<FConfig64, 1>>;

pub type Field64_2 = Fp2<F2Config64>;
pub struct F2Config64;
impl Fp2Config for F2Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = MontFp!("7");

    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        // Fq(7)**(((q^0) - 1) / 2)
        MontFp!("1"),
        // Fq(7)**(((q^1) - 1) / 2)
        MontFp!("18446744069414584320"),
    ];
}

pub type Field64_3 = Fp3<F3Config64>;
pub struct F3Config64;

impl Fp3Config for F3Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = MontFp!("2");

    const FROBENIUS_COEFF_FP3_C1: &'static [Self::Fp] = &[
        MontFp!("1"),
        // Fq(2)^(((q^1) - 1) / 3)
        MontFp!("4294967295"),
        // Fq(2)^(((q^2) - 1) / 3)
        MontFp!("18446744065119617025"),
    ];

    const FROBENIUS_COEFF_FP3_C2: &'static [Self::Fp] = &[
        MontFp!("1"),
        // Fq(2)^(((2q^1) - 2) / 3)
        MontFp!("18446744065119617025"),
        // Fq(2)^(((2q^2) - 2) / 3)
        MontFp!("4294967295"),
    ];

    // (q^3 - 1) = 2^32 * T where T = 1461501636310055817916238417282618014431694553085
    const TWO_ADICITY: u32 = 32;

    // 11^T
    const QUADRATIC_NONRESIDUE_TO_T: Fp3<Self> =
        Fp3::new(MontFp!("5944137876247729999"), MontFp!("0"), MontFp!("0"));

    // T - 1 / 2
    const TRACE_MINUS_ONE_DIV_TWO: &'static [u64] =
        &[0x80000002fffffffe, 0x80000002fffffffc, 0x7ffffffe];
}
