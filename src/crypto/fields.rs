use ark_ff::{Field, Fp128, Fp192, Fp64, MontBackend, MontConfig, PrimeField};

pub type Field256 = ark_test_curves::bls12_381::Fr;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FrConfig64;
pub type Field64 = Fp64<MontBackend<FrConfig64, 1>>;

#[derive(MontConfig)]
#[modulus = "340282366920938463463374557953744961537"]
#[generator = "3"]
pub struct FrConfig128;
pub type Field128 = Fp128<MontBackend<FrConfig128, 2>>;

#[derive(MontConfig)]
#[modulus = "4787605948707450321761805915146316350821882368518086721537"]
#[generator = "3"]
pub struct FrConfig192;
pub type Field192 = Fp192<MontBackend<FrConfig192, 3>>;

pub fn field_size_bits<F: Field>() -> usize {
    F::BasePrimeField::MODULUS_BIT_SIZE as usize * F::extension_degree() as usize
}
