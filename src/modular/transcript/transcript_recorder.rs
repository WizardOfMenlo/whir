use super::{Interaction, Transcript, TranscriptError, TranscriptPattern};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct TranscriptRecorder(TranscriptPattern);

impl TranscriptRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finalize(self) -> Result<TranscriptPattern, TranscriptError> {
        self.0.validate()?;
        Ok(self.0)
    }
}

impl Transcript for TranscriptRecorder {
    type Error = TranscriptError;

    fn interact(&mut self, interaction: Interaction) -> Result<(), Self::Error> {
        eprintln!("{interaction}");
        self.0.push(interaction)
    }
}
