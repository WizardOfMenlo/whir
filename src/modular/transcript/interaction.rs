use core::{
    any::{type_name, TypeId},
    fmt::Display,
};

/// A single abstract prover-verifier interaction
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Interaction {
    /// The kind of interaction.
    kind: InteractionKind,
    /// A label identifying the purpose of the value.
    label: &'static str,
    /// The type of the value.
    type_id: TypeId,
    /// The Rust name of the type of the value.
    type_name: &'static str,
    /// Length of the value.
    length: Length,
}

/// Kinds of prover-verifier interactions
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum InteractionKind {
    /// A message send in-band from prover to verifier.
    Message,
    /// A hint send out-of-band from prover to verifier.
    Hint,
    /// A challenge derived from the transform.
    Challenge,
    /// The start of a sub-protocol
    Begin,
    /// The end of a sub-protocol
    End,
}

/// Length of values involved in interactions.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Length {
    None,
    Scalar,
    Fixed(usize),
    Dynamic,
}

impl Interaction {
    pub fn new<T: 'static>(kind: InteractionKind, label: &'static str, length: Length) -> Self {
        let type_id = TypeId::of::<T>();
        let type_name = type_name::<T>();
        Self {
            kind,
            label,
            type_id,
            type_name,
            length,
        }
    }

    pub fn kind(&self) -> InteractionKind {
        self.kind
    }

    /// If it is an `InteractionKind::End`, return the corresponding `InteractionKind::Begin`
    pub(super) fn as_begin(self) -> Self {
        assert_eq!(self.kind, InteractionKind::End);
        Self {
            kind: InteractionKind::Begin,
            ..self
        }
    }
}

impl Display for Interaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.kind, self.label, self.type_name)
    }
}

impl Display for InteractionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message => write!(f, "MESSAGE"),
            Self::Hint => write!(f, "HINT"),
            Self::Challenge => write!(f, "CHALLENGE"),
            Self::Begin => write!(f, "BEGIN"),
            Self::End => write!(f, "END"),
        }
    }
}
