//! Allocation tracking for profiling memory usage, using tracing spans.
//!
//! Ported from [ProveKit](https://github.com/provekit/provekit)'s CLI profiler.
//!
//! # Usage
//!
//! 1. Enable the `alloc-track` feature (which also enables `tracing`).
//! 2. In your binary, set the global allocator and initialize the subscriber:
//!    ```rust,ignore
//!    #[global_allocator]
//!    static ALLOCATOR: whir::alloc_track::ProfilingAllocator =
//!        whir::alloc_track::ProfilingAllocator::new();
//!
//!    fn main() {
//!        whir::alloc_track::init_subscriber(&ALLOCATOR);
//!        // ...
//!    }
//!    ```
//! 3. Functions annotated with `#[tracing::instrument]` will automatically
//!    print memory stats (peak, local, allocation count) on span close.
//!
//! See `src/bin/alloc_report.rs` for a ready-to-run example.

use std::{
    alloc::{GlobalAlloc, Layout, System as SystemAlloc},
    cmp::max,
    fmt::{self, Display, Formatter, Write as _},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id},
    Level, Subscriber,
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

// ── Profiling Allocator ──────────────────────────────────────────────────

/// Custom allocator that keeps track of statistics to see program memory
/// consumption. Tracks current bytes, peak bytes, and allocation count.
pub struct ProfilingAllocator {
    /// Currently allocated bytes.
    current: AtomicUsize,
    /// Maximum allocated bytes reached so far.
    max: AtomicUsize,
    /// Total number of allocations done.
    count: AtomicUsize,
}

impl Default for ProfilingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingAllocator {
    pub const fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    pub fn current(&self) -> usize {
        self.current.load(Ordering::SeqCst)
    }

    pub fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    /// Reset peak to current and return the current value.
    pub fn reset_max(&self) -> usize {
        let current = self.current();
        self.max.store(current, Ordering::SeqCst);
        current
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

#[allow(unsafe_code)]
unsafe impl GlobalAlloc for ProfilingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = SystemAlloc.alloc(layout);
        let size = layout.size();
        let current = self
            .current
            .fetch_add(size, Ordering::SeqCst)
            .wrapping_add(size);
        self.max.fetch_max(current, Ordering::SeqCst);
        self.count.fetch_add(1, Ordering::SeqCst);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.current.fetch_sub(layout.size(), Ordering::SeqCst);
        SystemAlloc.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = SystemAlloc.alloc_zeroed(layout);
        let size = layout.size();
        let current = self
            .current
            .fetch_add(size, Ordering::SeqCst)
            .wrapping_add(size);
        self.max.fetch_max(current, Ordering::SeqCst);
        self.count.fetch_add(1, Ordering::SeqCst);
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let ptr = SystemAlloc.realloc(ptr, old_layout, new_size);
        let old_size = old_layout.size();
        if new_size > old_size {
            let diff = new_size - old_size;
            let current = self
                .current
                .fetch_add(diff, Ordering::SeqCst)
                .wrapping_add(diff);
            self.max.fetch_max(current, Ordering::SeqCst);
            self.count.fetch_add(1, Ordering::SeqCst);
        } else {
            self.current
                .fetch_sub(old_size - new_size, Ordering::SeqCst);
        }
        ptr
    }
}

// ── Span Stats Layer ─────────────────────────────────────────────────────

const DIM: &str = "\x1b[2m";
const UNDIM: &str = "\x1b[22m";

/// Extension data attached to each tracing span.
struct SpanData {
    depth: usize,
    time: Instant,
    memory: usize,
    allocations: usize,
    /// Peak memory is updated as it is not monotonic.
    peak_memory: usize,
    children: bool,
    kvs: Vec<(&'static str, String)>,
}

impl SpanData {
    fn new(attrs: &Attributes<'_>, depth: usize, allocator: &'static ProfilingAllocator) -> Self {
        let mut data = Self {
            depth,
            time: Instant::now(),
            memory: allocator.current(),
            allocations: allocator.count(),
            peak_memory: allocator.current(),
            children: false,
            kvs: Vec::new(),
        };
        attrs.record(&mut data);
        data
    }
}

impl Visit for SpanData {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.kvs.push((field.name(), format!("{value:?}")));
    }
}

struct FmtEvent<'a>(&'a mut String);

impl Visit for FmtEvent<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        match field.name() {
            "message" => {
                write!(self.0, " {value:?}").unwrap();
            }
            name => {
                write!(self.0, " {name}={value:?}").unwrap();
            }
        }
    }
}

/// Tracing subscriber layer that prints timing and memory stats for each span.
///
/// On span open, prints the span name with current memory.
/// On span close, prints duration, peak memory, local (own) memory, and
/// allocation count.
pub struct SpanStats {
    allocator: &'static ProfilingAllocator,
}

impl SpanStats {
    pub const fn new(allocator: &'static ProfilingAllocator) -> Self {
        Self { allocator }
    }
}

impl<S> Layer<S> for SpanStats
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    #[allow(clippy::significant_drop_tightening)]
    fn on_new_span(&self, attrs: &Attributes, id: &Id, ctx: Context<S>) {
        let span = ctx.span(id).expect("invalid span in on_new_span");

        // Update parent peak memory
        if let Some(parent) = span.parent() {
            if let Some(data) = parent.extensions_mut().get_mut::<SpanData>() {
                data.children = true;
                data.peak_memory = max(data.peak_memory, self.allocator.max());
            }
        }
        self.allocator.reset_max();

        // Attach SpanData if not already present
        if span.extensions().get::<SpanData>().is_none() {
            let depth = span.parent().map_or(0, |s| {
                s.extensions()
                    .get::<SpanData>()
                    .expect("parent span has no data")
                    .depth
                    + 1
            });
            let data = SpanData::new(attrs, depth, self.allocator);
            span.extensions_mut().insert(data);
        }

        // Print span open line
        let ext = span.extensions();
        let data = ext.get::<SpanData>().expect("span does not have data");

        let mut buffer = String::with_capacity(100);
        if data.depth >= 1 {
            for _ in 0..(data.depth - 1) {
                let _ = write!(&mut buffer, "│ ");
            }
            let _ = write!(&mut buffer, "├─");
        }
        let _ = write!(&mut buffer, "╮ ");
        let _ = write!(
            &mut buffer,
            "{DIM}{}::{UNDIM}{}",
            span.metadata().target(),
            span.metadata().name()
        );
        for (key, val) in &data.kvs {
            let _ = write!(&mut buffer, " {key}={val}");
        }
        let _ = write!(
            &mut buffer,
            " {DIM}start:{UNDIM} {}B{DIM} current, {UNDIM}{:#}{DIM} allocations{UNDIM}",
            human(self.allocator.current() as f64),
            human(self.allocator.count() as f64)
        );
        eprintln!("{buffer}");
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        let span = ctx.current_span().id().and_then(|id| ctx.span(id));

        let mut buffer = String::with_capacity(100);
        if let Some(span) = &span {
            if let Some(parent) = span.parent() {
                if let Some(data) = parent.extensions_mut().get_mut::<SpanData>() {
                    data.children = true;
                }
            }
            if let Some(data) = span.extensions().get::<SpanData>() {
                for _ in 0..=data.depth {
                    let _ = write!(&mut buffer, "│ ");
                }
                let elapsed = data.time.elapsed();
                let _ = write!(
                    &mut buffer,
                    "{DIM}{:6}s {UNDIM}",
                    human(elapsed.as_secs_f64())
                );
            }
        }

        match *event.metadata().level() {
            Level::TRACE => write!(&mut buffer, "TRACE"),
            Level::DEBUG => write!(&mut buffer, "DEBUG"),
            Level::INFO => write!(&mut buffer, "\x1b[1;32mINFO\x1b[0m"),
            Level::WARN => write!(&mut buffer, "\x1b[1;38;5;208mWARN\x1b[0m"),
            Level::ERROR => write!(&mut buffer, "\x1b[1;31mERROR\x1b[0m"),
        }
        .unwrap();

        let mut visitor = FmtEvent(&mut buffer);
        event.record(&mut visitor);
        eprintln!("{buffer}");
    }

    #[allow(clippy::significant_drop_tightening)]
    fn on_close(&self, id: Id, ctx: Context<S>) {
        let span = ctx.span(&id).expect("invalid span in on_close");
        let ext = span.extensions();
        let data = ext.get::<SpanData>().expect("span does not have data");
        let duration = data.time.elapsed();

        let mut buffer = String::with_capacity(100);
        if data.depth >= 1 {
            for _ in 0..(data.depth - 1) {
                let _ = write!(&mut buffer, "│ ");
            }
            let _ = write!(&mut buffer, "├─");
        }
        let _ = write!(&mut buffer, "╯ ");

        if data.children {
            let _ = write!(&mut buffer, "{DIM}{}: {UNDIM}", span.metadata().name());
        }

        let _ = write!(
            &mut buffer,
            "{}s{DIM} duration",
            human(duration.as_secs_f64()),
        );

        let peak_memory: usize = max(self.allocator.max(), data.peak_memory);
        let allocations = self.allocator.count() - data.allocations;
        let own = peak_memory.saturating_sub(data.memory);

        // Update parent peak memory
        if let Some(parent) = span.parent() {
            if let Some(pdata) = parent.extensions_mut().get_mut::<SpanData>() {
                pdata.peak_memory = max(pdata.peak_memory, peak_memory);
            }
        }

        let current_now = self.allocator.current();
        let _ = write!(
            &mut buffer,
            ", {UNDIM}{}B{DIM} peak memory, {UNDIM}{}B{DIM} local, {UNDIM}{}B{DIM} current, \
             {UNDIM}{:#}{DIM} allocations{UNDIM}",
            human(peak_memory as f64),
            human(own as f64),
            human(current_now as f64),
            human(allocations as f64)
        );

        eprintln!("{buffer}");
    }
}

// ── Human-readable number formatting ─────────────────────────────────────

/// Format a number with SI prefix (e.g., 1.23 M, 456 k, 789 n).
///
/// Uses the alternate flag `{:#}` to suppress the space before the suffix.
pub fn human(value: f64) -> impl Display {
    struct Human(f64);
    impl Display for Human {
        #[allow(clippy::cast_sign_loss)]
        fn fmt(&self, f: &mut Formatter) -> fmt::Result {
            let log10 = if self.0.is_normal() {
                self.0.abs().log10()
            } else {
                0.0
            };
            let si_power = ((log10 / 3.0).floor() as isize).clamp(-10, 10);
            let value = self.0 * 10_f64.powi((-si_power * 3) as i32);
            let digits =
                f.precision().unwrap_or(3) - 1 - 3.0f64.mul_add(-(si_power as f64), log10) as usize;
            let separator = if f.alternate() { "" } else { "\u{202F}" };
            if f.width() == Some(6) && digits == 0 {
                write!(f, " ")?;
            }
            write!(f, "{value:.digits$}{separator}")?;
            let suffix = "qryzafpnμm kMGTPEZYRQ"
                .chars()
                .nth((si_power + 10) as usize)
                .unwrap();
            if suffix != ' ' || f.width() == Some(6) {
                write!(f, "{suffix}")?;
            }
            Ok(())
        }
    }
    Human(value)
}

// ── Convenience initializer ──────────────────────────────────────────────

/// Initialize the global tracing subscriber with the [`SpanStats`] layer.
///
/// Call this once at the start of your binary, after setting
/// `#[global_allocator]`.
pub fn init_subscriber(allocator: &'static ProfilingAllocator) {
    use tracing::subscriber;
    use tracing_subscriber::{layer::SubscriberExt as _, Registry};

    let subscriber = Registry::default().with(SpanStats::new(allocator));
    subscriber::set_global_default(subscriber).expect("failed to set global tracing subscriber");
}
