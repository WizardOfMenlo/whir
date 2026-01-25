//! Traits and types for field morphisms.

use std::fmt::Debug;

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::type_info::{Type, TypeInfo, Typed};

/// Trait for a type representing a unital field homomorphism.
///
/// An `Embedding<Source = F, Target = G>` must satisfy
///
/// - `e.map(F::ONE) == G::ONE`
/// - `e.map(a + b) == e.map(a) + e.map(b)`
/// - `e.map(a * b) == e.map(a) * e.map(b)`
/// - `e.mixed_add(a, b) == a + e.map(b)`
/// - `e.mixed_mul(a, b) == a * e.map(b)`
///
pub trait Embedding: Clone + Debug + TypeInfo + Serialize + for<'de> Deserialize<'de> {
    type Source: Field;
    type Target: Field;

    fn map(&self, dom: Self::Source) -> Self::Target;

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod * self.map(dom)
    }

    #[inline]
    fn mixed_add(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod + self.map(dom)
    }
}

/// The identiy embedding of a field in into itself.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "F: Field")]
pub struct Identity<F: Field> {
    field: Type<F>,
}

/// The basefield embedding morphism.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "F: Field")]
pub struct Basefield<F: Field> {
    extension: Type<F>,
}

/// The Frobenius automorphism.
///
/// Mostly for fun, it serves no purpose in whir.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "F: Field")]
pub struct Frobenius<F: Field> {
    field: Type<F>,
    power: u64,
}

/// The composition of two morphisms.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "A: Embedding, B: Embedding<Source = A::Target>")]
pub struct Compose<A: Embedding, B: Embedding<Source = A::Target>> {
    inner: Typed<A>,
    outer: Typed<B>,
}

impl<F: Field> Identity<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Field> Basefield<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Field> Frobenius<F> {
    pub fn new(power: u64) -> Self {
        Self {
            field: Type::new(),
            power: power % F::extension_degree(),
        }
    }
}

impl<A, B> Compose<A, B>
where
    A: Embedding,
    B: Embedding<Source = A::Target>,
{
    pub fn new(inner: A, outer: B) -> Self {
        Self {
            inner: Typed::new(inner),
            outer: Typed::new(outer),
        }
    }
}

impl<F: Field> Embedding for Identity<F> {
    type Source = F;
    type Target = F;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        dom
    }
}

impl<F: Field> Embedding for Basefield<F> {
    type Source = F::BasePrimeField;
    type Target = F;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        Self::Target::from_base_prime_field(dom)
    }

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod.mul_by_base_prime_field(&dom)
    }
}

impl<F: Field> Embedding for Frobenius<F> {
    type Source = F;
    type Target = F;

    #[inline]
    fn map(&self, mut dom: Self::Source) -> Self::Target {
        dom.frobenius_map_in_place(self.power as usize);
        dom
    }
}

impl<A, B> Embedding for Compose<A, B>
where
    A: Embedding,
    B: Embedding<Source = A::Target>,
{
    type Source = A::Source;
    type Target = B::Target;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        self.outer.map(self.inner.map(dom))
    }

    #[inline]
    fn mixed_add(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        self.outer.mixed_add(cod, self.inner.map(dom))
    }

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        self.outer.mixed_mul(cod, self.inner.map(dom))
    }
}

impl<F: Field> TypeInfo for Identity<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "identity".into()
    }
}

impl<F: Field> TypeInfo for Basefield<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "basefield".into()
    }
}

impl<F: Field> TypeInfo for Frobenius<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "frobenius".into()
    }
}

impl<A: Embedding, B: Embedding<Source = A::Target>> TypeInfo for Compose<A, B> {
    type Info = String;

    fn type_info() -> Self::Info {
        "compose".into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_ff::{AdditiveGroup, Field};
    use proptest::{prelude::Strategy, proptest};

    use crate::crypto::embedding::Embedding;

    pub fn test_embedding<E: Embedding>(
        e: E,
        source: &impl Strategy<Value = E::Source>,
        target: &impl Strategy<Value = E::Target>,
    ) {
        assert_eq!(e.map(E::Source::ZERO), E::Target::ZERO);
        assert_eq!(e.map(E::Source::ONE), E::Target::ONE);
        proptest!(|(a in source, b in source)| {
            assert_eq!(e.map(a) + e.map(b), e.map(a + b));
        });
        proptest!(|(a in source, b in source)| {
            assert_eq!(e.map(a) * e.map(b), e.map(a * b));
        });
        proptest!(|(a in target, b in source)| {
            assert_eq!(e.mixed_add(a, b), a + e.map(b));
        });
        proptest!(|(a in target, b in source)| {
            assert_eq!(e.mixed_mul(a, b), a * e.map(b));
        });
    }
}
