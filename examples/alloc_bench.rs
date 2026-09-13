//! Before/after: allocation churn and wall time for the same parallel
//! workload under three strategies.
//!
//!   1. **fresh** — `vec![0.0; N]` per task (the pre-pool code);
//!   2. **pooled** — [`BufferPool`] leases (this crate);
//!   3. **map_init** — rayon's native per-worker scratch
//!      (`map_init(|| ..., |scratch, item| ...)`), the no-pool alternative
//!      when one loop owns all its scratch. Note it re-runs its initializer
//!      once per worker *per invocation* of the parallel loop — a loop called
//!      repeatedly (every SCF iteration, every Hessian row, ...) still pays
//!      that churn, while pool contents persist across calls.
//!
//! Buffers are 2 MiB — the shape of real numerical scratch (GEMM workspaces,
//! grid-chunk buffers). In this regime per-task allocation (zero-fill, and
//! `mmap`/page-fault traffic for sizes beyond the allocator's small-chunk
//! caches) is a visible fraction of each task, and pooling recovers it; the
//! allocation *count* drops from thousands to roughly the thread count.
//!
//! Two honest caveats, visible if you shrink `N` to cache-sized buffers and
//! single-pass work:
//!
//!   * the lease/unlease lock pair is ~100 ns — it only matters for tasks in
//!     the low microseconds (then prefer `map_init`, or coarsen tasks);
//!   * a *recycled* buffer's cache lines live on the previous owner's core,
//!     so tiny hot-in-L2 buffers pay a cross-core transfer that a freshly
//!     zeroed (same-thread) buffer does not — at real scratch sizes, data is
//!     re-streamed from memory anyway and the effect vanishes.
//!
//! Numbers vary by machine and allocator; the qualitative ordering is stable.
//!
//! Run with `cargo run --release --example alloc_bench`.

use std::hint::black_box;
use std::time::Instant;

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const N: usize = 262_144; // f64 elements per buffer = 2 MiB
const TASKS: usize = 512;
const ROUNDS: usize = 4; // measured reps of the whole workload run REPS times
const PASSES: usize = 1; // passes over the buffer per lease (1st overwrites)
const REPS: usize = 7;

fn work(buf: &mut [f64], task: usize) -> f64 {
    // Stand-in for real scratch traffic: the first pass fully overwrites the
    // buffer (like a beta = 0 GEMM), the rest read-modify-write it, like a
    // workspace touched repeatedly between checkout and return.
    let mut acc = 0.0;
    for pass in 0..PASSES {
        let (scale, bias) = if pass == 0 {
            (0.0, 1.0)
        } else {
            (0.999, 0.001)
        };
        for (i, x) in buf.iter_mut().enumerate() {
            *x = *x * scale + ((i + task * 7 + pass) % 1024) as f64 * bias;
            acc += *x;
        }
    }
    acc
}

fn timed(name: &str, f: impl Fn() -> f64) {
    // Best-of-N: this example's purpose is the *ordering* of strategies, and
    // single wall-clock runs on a busy multi-core box flip orderings; the
    // minimum over several reps is the usual robust statistic.
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    for _ in 0..REPS {
        let t0 = Instant::now();
        checksum = black_box(f());
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "{name:>10}: {:>8.1} ms (best of {REPS})  checksum {checksum:.0}",
        best * 1e3
    );
}

fn main() {
    let pool = BufferPool::new(|| vec![0.0f64; N]);

    timed("fresh", || {
        (0..ROUNDS)
            .map(|r| {
                (0..TASKS)
                    .into_par_iter()
                    .map(|task| {
                        let mut buf = vec![0.0; N]; // malloc + zero, per task
                        black_box(work(&mut buf, task + r))
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    });

    timed("pooled", || {
        (0..ROUNDS)
            .map(|r| {
                (0..TASKS)
                    .into_par_iter()
                    .map(|task| {
                        let mut buf = pool.get(); // recycled after warm-up
                        black_box(work(&mut buf, task + r))
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    });

    timed("map_init", || {
        (0..ROUNDS)
            .map(|r| {
                (0..TASKS)
                    .into_par_iter()
                    .map_init(
                        || vec![0.0; N], // one scratch per rayon worker
                        |buf, task| black_box(work(buf, task + r)),
                    )
                    .sum::<f64>()
            })
            .sum::<f64>()
    });

    let stats = pool.stats();
    println!(
        "\npooled stats: {} leases, {} allocations, {} reuses \
         (fresh performed {} big allocations for the same work)",
        stats.leases,
        stats.allocations,
        stats.reuses(),
        ROUNDS * TASKS
    );
}
