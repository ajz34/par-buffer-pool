//! Variable-size scratch: one grow-only pool, and size-class pools.
//!
//! A pool is size-blind: it recycles whatever buffer it was handed, whatever
//! the next lease needs. Workloads whose buffer sizes vary have two standard
//! answers, both built on `Vec`'s split between *length* (logical) and
//! *capacity* (physical):
//!
//! 1. **Grow-only pool** (simplest, usually best): every buffer starts as an
//!    empty `Vec`, and every lease does `clear()` + `resize(n, 0.0)`.
//!    Buffers grow to the high-water mark of demand and keep that capacity,
//!    so all sizes share one pool; a smaller lease just uses a prefix.
//!
//! 2. **Size-class pools**: one pool per power-of-two capacity class, so a
//!    task asking for 16 elements never ties up a 4096-element buffer.
//!    Useful when size variance is extreme.
//!
//! Run with `cargo run --release --example size_buckets`.

use par_buffer_pool::{BufferPool, SharedPooled};
use rayon::prelude::*;

type Lease = SharedPooled<'static, Vec<f64>>;

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
                // Capacity only; contents are set per lease.
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

    println!(
        "{name}: {total:.1} over {} tasks of {}..={} elements",
        sizes.len(),
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap()
    );
}

fn main() {
    // Task sizes in 16..=4111 from a small linear congruential generator.
    let mut seed = 1u32;
    let sizes: Vec<usize> = (0..512)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            16 + (seed as usize) % 4096
        })
        .collect();

    let grow = AnySizeScratch::new();
    run("grow-only", &sizes, &|n| grow.get(n));
    println!(
        "  -> {} allocations for {} leases (rest were capacity reuses)",
        grow.pool.stats().allocations,
        grow.pool.stats().leases
    );

    let classed = ClassScratch::new(*sizes.iter().max().unwrap());
    run("size-class", &sizes, &|n| classed.get(n));
    let stats: Vec<_> = classed.pools.iter().map(|p| p.stats()).collect();
    let allocations: usize = stats.iter().map(|s| s.allocations).sum();
    let leases: usize = stats.iter().map(|s| s.leases).sum();
    println!("  -> {allocations} allocations for {leases} leases");
}
