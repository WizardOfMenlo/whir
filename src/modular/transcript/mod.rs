mod interaction;
mod transcript_pattern;
mod transcript_player;
mod transcript_recorder;

use core::error::Error;

pub use self::{
    interaction::{Interaction, InteractionKind, Length},
    transcript_pattern::{TranscriptError, TranscriptPattern},
    transcript_player::TranscriptPlayer,
    transcript_recorder::TranscriptRecorder,
};

pub trait Transcript {
    type Error: Error;

    fn interact(&mut self, interaction: Interaction) -> Result<(), Self::Error>;
}

pub trait TranscriptExt {
    type Error: Error;

    fn message<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;
    fn message_array<T: 'static>(
        &mut self,
        label: &'static str,
        length: usize,
    ) -> Result<(), Self::Error>;
    fn message_slice<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;

    fn hint<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;

    fn challenge<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;

    fn begin<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;
    fn end<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error>;
}

impl<Tr: Transcript> TranscriptExt for Tr {
    type Error = Tr::Error;

    fn message<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Message,
            label,
            Length::Scalar,
        ))
    }

    fn message_array<T: 'static>(
        &mut self,
        label: &'static str,
        length: usize,
    ) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Message,
            label,
            Length::Fixed(length),
        ))
    }

    fn message_slice<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Message,
            label,
            Length::Dynamic,
        ))
    }

    fn hint<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Hint,
            label,
            Length::Scalar,
        ))
    }

    fn challenge<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Challenge,
            label,
            Length::Scalar,
        ))
    }

    fn begin<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::Begin,
            label,
            Length::None,
        ))
    }

    fn end<T: 'static>(&mut self, label: &'static str) -> Result<(), Self::Error> {
        self.interact(Interaction::new::<T>(
            InteractionKind::End,
            label,
            Length::None,
        ))
    }
}
