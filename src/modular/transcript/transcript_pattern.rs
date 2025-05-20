use core::fmt::Display;

use thiserror::Error;

use super::{Interaction, InteractionKind};

/// Abstract transcript containing prover-verifier interactions
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct TranscriptPattern {
    interactions: Vec<Interaction>,
}

/// Errors when validating a transcript.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Error)]
pub enum TranscriptError {
    #[error("Missing BEGIN for {end} at {position}")]
    MissingBegin {
        position: usize,
        end: Box<Interaction>,
    },
    #[error("Mismatch {begin} at {begin_position} for {end} at {end_position}")]
    MismatchedBeginEnd {
        begin_position: usize,
        begin: Box<Interaction>,
        end_position: usize,
        end: Box<Interaction>,
    },
    #[error("Missing END for {begin} at {position}")]
    MissingEnd {
        position: usize,
        begin: Box<Interaction>,
    },
}

impl TranscriptPattern {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn interactions(&self) -> &[Interaction] {
        &self.interactions
    }

    /// Make sure [`InteractionKind::Begin`] and [`InteractionKind::End`] match.
    pub fn validate(&self) -> Result<(), TranscriptError> {
        let mut stack = Vec::new();
        for (position, &interaction) in self.interactions.iter().enumerate() {
            match interaction.kind() {
                InteractionKind::Begin => stack.push((position, interaction)),
                InteractionKind::End => {
                    let Some((position, begin)) = stack.pop() else {
                        return Err(TranscriptError::MissingBegin {
                            position,
                            end: Box::new(interaction),
                        });
                    };
                    let expected = interaction.as_begin();
                    if begin != expected {
                        return Err(TranscriptError::MismatchedBeginEnd {
                            begin_position: position,
                            begin: Box::new(begin),
                            end_position: self.interactions.len(),
                            end: Box::new(interaction),
                        });
                    }
                }
                _ => {}
            }
        }
        if let Some((position, begin)) = stack.pop() {
            return Err(TranscriptError::MissingEnd {
                position,
                begin: Box::new(begin),
            });
        }
        Ok(())
    }

    pub(super) fn push(&mut self, interaction: Interaction) -> Result<(), TranscriptError> {
        if interaction.kind() == InteractionKind::End {
            let begin = self.last_open_begin();
            let Some((position, begin)) = begin else {
                return Err(TranscriptError::MissingBegin {
                    position: self.interactions.len(),
                    end: Box::new(interaction),
                });
            };
            let expected = interaction.as_begin();
            if expected != begin {
                return Err(TranscriptError::MismatchedBeginEnd {
                    begin_position: position,
                    begin: Box::new(begin),
                    end_position: self.interactions.len(),
                    end: Box::new(interaction),
                });
            }
        }
        self.interactions.push(interaction);
        Ok(())
    }

    /// Return the last unclosed BEGIN interaction.
    fn last_open_begin(&self) -> Option<(usize, Interaction)> {
        // Reverse search to find matching begin
        let mut stack = 0;
        for (position, &interaction) in self.interactions.iter().rev().enumerate() {
            match interaction.kind() {
                InteractionKind::End => stack += 1,
                InteractionKind::Begin => {
                    if stack == 0 {
                        return Some((position, interaction));
                    }
                    stack -= 1;
                }
                _ => {}
            }
        }
        None
    }
}

impl Display for TranscriptPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut indentation = 0;
        for (position, interaction) in self.interactions.iter().enumerate() {
            write!(f, "{position:>4} ")?;
            if interaction.kind() == InteractionKind::End {
                indentation -= 1;
            }
            for _ in 0..indentation {
                write!(f, "  ")?;
            }
            writeln!(f, "{interaction}")?;
            if interaction.kind() == InteractionKind::Begin {
                indentation += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        dbg!(size_of::<Interaction>());
        dbg!(size_of::<TranscriptError>());
    }
}
