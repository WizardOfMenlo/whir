//! Allocation tracking for profiling memory usage at each protocol step.
//!
//! # Usage
//!
//! 1. Enable the `alloc-track` feature.
//! 2. In your binary, set the global allocator:
//!    ```rust,ignore
//!    #[global_allocator]
//!    static ALLOC: whir::alloc_track::TrackingAllocator = whir::alloc_track::TrackingAllocator;
//!    ```
//! 3. Run your binary — each instrumented protocol step will print allocation
//!    counts and bytes to stderr.
//!
//! See `src/bin/alloc_report.rs` for a ready-to-run example.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// ── Global counters ──────────────────────────────────────────────────────

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// A global allocator that counts every allocation.
///
/// Wraps `std::alloc::System` and atomically increments counters on each
/// `alloc` / `realloc`.  De-allocations are forwarded without counting
/// (tracking freed memory is possible but adds overhead and isn't needed
/// for allocation-count profiling).
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

// ── Snapshot API ─────────────────────────────────────────────────────────

/// A point-in-time snapshot of allocation counters.
///
/// Take a snapshot before a code region, then call [`Snapshot::elapsed`]
/// to see how many allocations occurred in between.
#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub allocs: u64,
    pub bytes: u64,
}

/// The delta between two snapshots.
#[derive(Debug, Clone, Copy)]
pub struct AllocDelta {
    pub allocs: u64,
    pub bytes: u64,
}

impl Snapshot {
    /// Capture the current allocation counters.
    #[inline]
    pub fn now() -> Self {
        Self {
            allocs: ALLOC_COUNT.load(Ordering::Relaxed),
            bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    /// Compute how many allocations have happened since this snapshot.
    #[inline]
    pub fn elapsed(&self) -> AllocDelta {
        let now = Self::now();
        AllocDelta {
            allocs: now.allocs.wrapping_sub(self.allocs),
            bytes: now.bytes.wrapping_sub(self.bytes),
        }
    }
}

impl std::fmt::Display for AllocDelta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>8} allocs, {:>10}",
            self.allocs,
            format_bytes(self.bytes)
        )
    }
}

/// Reset all counters to zero (useful before a profiling run).
pub fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

/// Print a labelled allocation-delta line to stderr.
pub fn report(label: &str, snap: &Snapshot) {
    let delta = snap.elapsed();
    eprintln!("  {label:50} {delta}");
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

// ── Macros for zero-cost instrumentation ─────────────────────────────────

/// Take an allocation snapshot. Expands to `()` when `alloc-track` is off.
#[macro_export]
macro_rules! alloc_snap {
    () => {{
        $crate::alloc_track::Snapshot::now()
    }};
}

/// Print an allocation report line and reset the snapshot variable.
///
/// Usage: `alloc_report!("label", snap);`
///
/// After this macro, `snap` holds a fresh snapshot for the next region.
#[macro_export]
macro_rules! alloc_report {
    ($label:expr, $snap:ident) => {{
        $crate::alloc_track::report($label, &$snap);
        $snap = $crate::alloc_track::Snapshot::now();
    }};
}
