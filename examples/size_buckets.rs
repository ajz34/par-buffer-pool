//! Variable-size scratch: one grow-only pool, and size-class pools.
//!
//! Real workloads do not always want the same buffer length, and a pool is
//! size-blind: it recycles whatever it was given. Two standard answers, both
//! built on `Vec`'s split between *length* (logical) and *capacity*
//! (physical):
//!
//! 1. **Grow-only pool** (simplest, usually best): every buffer is a
//!    `Vec` started from `Vec::new`, and every lease does `clear()` +
//!    `resize(n, 0.0)`. Buffers grow to the high-water mark of demand and
//!    keep that capacity, so all sizes share the same pool; a smaller lease
//!    just uses a prefix.
//!
//! 2. **Size-class pools**: one pool per power-of-two capacity class, so a
//!    task asking for 16 elements never ties up a 4096-element buffer until
//!    the next big task comes. Useful when size variance is extreme.
//!
//! Run with `cargo run --release --example size_buckets`.

use par_buffer_pool::{BufferPool, Pooled};
use rayon::prelude::*;

type Lease = Pooled<'static, Vec<f64>>;

/// One grow-only pool serving any requested length.
struct AnySizeScratch {
    pool: BufferPool<'static, Vec<f64>>,
}

impl AnySizeScratch {
    fn new() -> Self {
        // The closure captures nothing, so it is 'static and this pool can
        // live in a long-lived driver struct.
        AnySizeScratch {
            pool: BufferPool::new(Vec::new),
        }
    }

    /// Lease a zeroed buffer of exactly `n` elements; capacity is recycled
    /// across leases of any size, so only growth ever allocates.
    fn get(&self, n: usize) -> Lease {
        let mut buf = self.pool.get();
        buf.clear();
        buf.resize(n, 0.0); // allocation-free whenever capacity >= n
        buf
    }
}

/// Pools keyed by power-of-two size class, for extreme size variance.
struct ClassScratch {
    pools: Vec<BufferPool<'static, Vec<f64>>>,
}

impl ClassScratch {
    /// Builds pools for classes 16..=next_power_of_two(max_n) elements.
    fn new(max_n: usize) -> Self {
        let max_class = max_n.next_power_of_two().trailing_zeros();
        let pools = (4..=max_class)
            .map(|c| {
                let cap = 1usize << c;
                // Uninitialized capacity: contents are set per lease.
                BufferPool::new(move || Vec::with_capacity(cap))
            })
            .collect();
        ClassScratch { pools }
    }

    fn get(&self, n: usize) -> Lease {
        let class = n.max(1).next_power_of_two().trailing_zeros() as usize;
        let mut buf = self.pools[class - 4].get();
        buf.clear();
        buf.resize(n, 0.0);
        buf
    }
}

fn run(name: &str, sizes: &[usize], scratch: &(dyn Fn(usize) -> Lease + Send + Sync)) {
    let total: f64 = sizes
        .par_iter()
        .map(|&n| {
            let mut buf = scratch(n);
            let mut acc = 0.0;
            for (k, b) in buf.iter_mut().enumerate() {
                *b = (k % 17) as f64 * 0.5;
                acc += *b;
            }
            acc
        })
        .sum();

    let total_requested: usize = sizes.iter().sum();
    println!(
        "{name}: {total:.1} over {} tasks, {} elements requested in total",
        sizes.len(),
        total_requested
    );
}

fn main() {
    // Deterministic pseudo-random task sizes spanning two orders of magnitude.
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let sizes: Vec<usize> = (0..512).map(|_| 16 + (next() % 4096) as usize).collect();

    let grow = AnySizeScratch::new();
    run("grow-only", &sizes, &|n| grow.get(n));
    println!(
        "  -> {} allocations for {} leases (rest were capacity reuses)",
        grow.pool.stats().allocations,
        grow.pool.stats().leases
    );

    let classed = ClassScratch::new(sizes.iter().copied().max().unwrap());
    run("size-class", &sizes, &|n| classed.get(n));
    let stats: Vec<_> = classed.pools.iter().map(|p| p.stats()).collect();
    let allocations: usize = stats.iter().map(|s| s.allocations).sum();
    let leases: usize = stats.iter().map(|s| s.leases).sum();
    println!("  -> {allocations} allocations for {leases} leases");
}
