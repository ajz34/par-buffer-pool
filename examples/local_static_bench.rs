//! `BufferPool` vs. `ThreadLocalPool` vs. no pool: where each wins.
//!
//! The two pools of this crate differ in one thing only: where the parked
//! buffers live and how a lease gets one back. This benchmark measures that
//! difference directly, against both baselines:
//!
//! ```text
//! strategy   | per-task lease/return mechanism
//! -----------+-------------------------------------------------------------
//! fresh      | malloc + zero-fill (no pool)
//! pooled     | BufferPool: shared-mutex lock/unlock pair + Vec push/pop
//! local      | ThreadLocalPool::get: TLS access + Option take/put (no lock)
//! local_with | ThreadLocalPool::with: the same path, closure-shaped
//! hoisted    | plain stack local reused across the loop (the floor)
//! map_init   | rayon's built-in per-worker scratch (no sync)
//! ```
//!
//! Two regimes are measured:
//!
//!   * *lease cost* — single-threaded, near-zero work per lease, several
//!     buffer sizes: isolates the mechanism numbers above as ns/lease.
//!   * *parallel* — rayon with real per-task work: 64 KiB tasks (lock
//!     contention and cross-core buffer reuse are visible) and 2 MiB tasks
//!     (allocation-dominated; every reuse strategy wins by the same margin).
//!
//! Honest caveats:
//!
//!   * `ThreadLocalPool` initializers must be `'static` and each thread
//!     parks at most one buffer per pool — features, not speed, and not
//!     measured here. A guard dropped on another thread migrates the buffer
//!     to that thread's slot; a shared pool instead returns it to any
//!     thread's next lease.
//!   * The shared pool hands out *recycled* buffers whose cache lines may
//!     live on another core. With tiny hot buffers that — plus the shared
//!     mutex — is the real cost being measured; for big buffers it vanishes
//!     into noise.
//!   * This example builds with the `stats` feature on (the crate's
//!     dev-dependency enables it for examples/tests), which adds one
//!     relaxed add on a shared cache line per lease to *both* pools. For
//!     default-feature numbers, copy this file into a crate that depends on
//!     `par-buffer-pool` without the feature. Reference numbers measured on
//!     the author's machine (16-core Ryzen 9 9955HX, glibc):
//!     lease cost — pooled ≈ 26 ns (30 with `stats`), local ≈ 9 ns (10
//!     with `stats`), hoisted floor ≈ 2.5 ns, fresh 16-305 ns by size;
//!     parallel ~4.5 µs tasks at 16 workers — fresh 14.8 / local 14.1 /
//!     map_init 14.1 / pooled 25.9 ms (stats off): the pooled 2x is the
//!     shared mutex's futex convoy, and it disappears with fewer threads
//!     or with tasks ≳ 10 µs.
//!
//! Run with `cargo run --release --example local_static_bench`.

use std::hint::black_box;
use std::time::Instant;

use par_buffer_pool::{BufferPool, ThreadLocalPool};
use rayon::prelude::*;

const REPS: usize = 5; // measured repetitions; best-of-N is reported

/// Stand-in for real scratch traffic: one overwrite pass, then a reduce.
/// (Same workload as `alloc_bench`.)
fn work(buf: &mut [f64], task: usize) -> f64 {
    let mut acc = 0.0;
    for (i, x) in buf.iter_mut().enumerate() {
        *x = ((i + task * 7) % 1024) as f64;
        acc += *x;
    }
    acc
}

/// Near-zero work: touch three cache lines, enough to keep the buffer alive.
fn touch(buf: &mut [f64], task: usize) -> f64 {
    buf[0] = task as f64;
    buf[buf.len() / 2] += 1.0;
    buf[buf.len() - 1] += 2.0;
    buf[0] + buf[buf.len() / 2] + buf[buf.len() - 1]
}

/// Best-of-N wall time in seconds; the checksum defeats dead-code elimination.
fn best_of(f: impl Fn() -> f64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        black_box(f());
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

/// Part 1: mechanism cost of getting a buffer in and out, ~no work per lease.
fn lease_cost(n: usize, iters: usize) {
    println!(
        "\n--- lease cost, single thread: {} leases of {:.0} KiB ---",
        iters,
        n * 8 / 1024
    );

    let pool = BufferPool::new(|| vec![0.0f64; n]);
    let local = ThreadLocalPool::new(move || vec![0.0f64; n]);

    let fresh = best_of(|| {
        let mut acc = 0.0;
        for task in 0..iters {
            let mut buf = vec![0.0; n]; // malloc + zero, per lease
            acc += black_box(touch(&mut buf, task));
        }
        acc
    });
    let pooled = best_of(|| {
        let mut acc = 0.0;
        for task in 0..iters {
            let mut buf = pool.get(); // lock/pop, then lock/push on drop
            acc += black_box(touch(&mut buf, task));
        }
        acc
    });
    let local_get = best_of(|| {
        let mut acc = 0.0;
        for task in 0..iters {
            let mut buf = local.get(); // TLS take, TLS put on drop
            acc += black_box(touch(&mut buf, task));
        }
        acc
    });
    let local_with = best_of(|| {
        let mut acc = 0.0;
        for task in 0..iters {
            acc += local.with(|buf| black_box(touch(buf, task)));
        }
        acc
    });
    let hoisted = best_of(|| {
        let mut buf = vec![0.0; n]; // the floor: no mechanism at all
        let mut acc = 0.0;
        for task in 0..iters {
            acc += black_box(touch(&mut buf, task));
        }
        acc
    });

    for (name, secs) in [
        ("fresh", fresh),
        ("pooled", pooled),
        ("local", local_get),
        ("local_with", local_with),
        ("hoisted", hoisted),
    ] {
        println!("{name:>10}: {:>6.1} ns/lease", secs * 1e9 / iters as f64);
    }
}

/// Part 2: rayon throughput with real per-task work, `n` elements per buffer.
fn parallel(n: usize, tasks: usize, rounds: usize) {
    println!(
        "\n--- parallel (rayon): {tasks} tasks x {rounds} rounds, {:.0} KiB/buffer ---",
        n * 8 / 1024
    );

    let pool = BufferPool::new(|| vec![0.0f64; n]);
    let local = ThreadLocalPool::new(move || vec![0.0f64; n]);

    let fresh = best_of(|| {
        (0..rounds)
            .map(|_| {
                (0..tasks)
                    .into_par_iter()
                    .map(|task| {
                        let mut buf = vec![0.0; n];
                        black_box(work(&mut buf, task))
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    });
    let pooled = best_of(|| {
        (0..rounds)
            .map(|_| {
                (0..tasks)
                    .into_par_iter()
                    .map(|task| {
                        let mut buf = pool.get();
                        black_box(work(&mut buf, task))
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    });
    let local_get = best_of(|| {
        (0..rounds)
            .map(|_| {
                (0..tasks)
                    .into_par_iter()
                    .map(|task| {
                        let mut buf = local.get();
                        black_box(work(&mut buf, task))
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
    });
    let local_with = best_of(|| {
        (0..rounds)
            .map(|_| {
                (0..tasks)
                    .into_par_iter()
                    .map(|task| local.with(|buf| black_box(work(buf, task))))
                    .sum::<f64>()
            })
            .sum::<f64>()
    });
    let map_init = best_of(|| {
        (0..rounds)
            .map(|_| {
                (0..tasks)
                    .into_par_iter()
                    .map_init(
                        || vec![0.0; n],
                        |buf, task| black_box(work(buf, task)),
                    )
                    .sum::<f64>()
            })
            .sum::<f64>()
    });

    for (name, secs) in [
        ("fresh", fresh),
        ("pooled", pooled),
        ("local", local_get),
        ("local_with", local_with),
        ("map_init", map_init),
    ] {
        println!("{name:>10}: {:>8.1} ms (best of {REPS})", secs * 1e3);
    }

    let stats = pool.stats();
    let local_stats = local.stats();
    println!(
        "allocations for {tasks}x{rounds} tasks — pooled: {}, local: {}",
        stats.allocations, local_stats.allocations,
    );
}

fn main() {
    // Small-to-mid buffers: here the lease mechanism is a visible fraction of
    // the per-lease cost, so fresh/pooled/local actually separate.
    lease_cost(128, 400_000); // 1 KiB
    lease_cost(1024, 50_000); // 8 KiB
    lease_cost(8192, 6_000); // 64 KiB

    // Tiny parallel tasks: ~µs of work each, so per-task lease overhead and
    // lock contention across 16 workers matter.
    parallel(8192, 50_000, 1);
    // Huge parallel tasks: allocation dominates; all reuse strategies tie.
    parallel(262_144, 512, 2);
}
