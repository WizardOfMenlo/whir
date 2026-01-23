pub mod ark_serde;
pub mod bits;
pub mod cmdline_utils;
pub mod crypto; // Crypto utils
pub mod domain; // Domain that we are evaluating over
pub mod hash;
pub mod ntt;
pub mod parameters;
pub mod poly_utils; // Utils for polynomials
pub mod protocols;
pub mod transcript;
pub mod type_info;
pub mod type_map;
pub mod utils; // Utils in general
pub mod whir; // The real prover

#[cfg(test)]
mod tests {
    use std::sync::Once;

    pub fn init() {
        static INIT: Once = Once::new();

        #[cfg(not(feature = "tracing"))]
        INIT.call_once(|| {});

        #[cfg(feature = "tracing")]
        INIT.call_once(|| {
            use tracing_subscriber::{fmt, fmt::format::FmtSpan, EnvFilter};

            // Respect RUST_LOG if set, otherwise default for tests.
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

            // Create a writer compatible with the testing harnesses
            fmt()
                .with_env_filter(filter)
                .with_span_events(FmtSpan::ENTER)
                .with_test_writer()
                .init();

            tracing::debug!("Initialized test logger");
        });
    }
}
