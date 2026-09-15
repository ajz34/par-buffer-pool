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
//! pooled     | BufferPool: per-thread shard lock/unlock + Vec push/pop
//! local      | ThreadLocalPool::get: TLS access + Option take/put (no lock)
//! local_with | ThreadLocalPool::with: the same path, closure-shaped
//! hoisted    | plain stack local reused across the loop (the floor)
//! map_init   | rayon's built-in per-worker scratch (no sync)
//! ```
//!
//! Two regimes are measured:
//!
//!   * *lease cost* — single-threaded, near-zero work per lease, several
//!     buffer sizes: isolates the mechanisms above, one lease at a time.
//!   * *parallel* — rayon with real per-task work: 64 KiB tasks (recycled
//!     buffers' cross-core cache lines are visible) and 2 MiB tasks
//!     (allocation-dominated; every reuse strategy ties).
//!
//! Honest caveats:
//!
//!   * `ThreadLocalPool` initializers must be `'static` and each thread
//!     parks at most one buffer per pool — features, not speed, and not
//!     measured here. A guard dropped on another thread migrates the buffer
//!     to that thread's slot; a shared pool instead returns it to any
//!     thread's next lease.
//!   * The shared pool hands out *recycled* buffers whose cache lines may
//!     live on another core. With tiny hot buffers that — plus the shard
//!     lock round trip — is the real cost being measured; for big buffers
//!     it vanishes into noise.
//!   * This example builds with the `stats` feature on (the crate's
//!     dev-dependency enables it for examples/tests), which adds one
//!     relaxed add on a shared cache line per lease to *both* pools — about
//!     one cycle, already included below. For default-feature numbers, copy
//!     this file into a crate that depends on `par-buffer-pool` without the
//!     feature. Reference numbers, measured on an AMD Ryzen 9 9950X3D
//!     (16 cores / 32 hardware threads, Linux, glibc, rustc 1.97) at 16
//!     rayon workers (`RAYON_NUM_THREADS=16`), in CPU cycles at that chip's
//!     ≈ 5.7 GHz boost — re-measure on your hardware before trusting
//!     orderings: lease cost — pooled ≈ 140 cycles, local ≈ 60-75, hoisted
//!     floor ≈ 13, fresh ≈ 110 cycles at 1 KiB rising to ≈ 1500-2000 at
//!     64 KiB. Parallel, 64 KiB tasks at 16 workers: the strategies tie —
//!     fresh 14.2 / pooled 13.2 / local 13.7 / map_init 13.4 ms for 50k
//!     tasks (best of 5; run-to-run spread a few percent). Real per-task
//!     work hides the shard lock: a one-pile pool convoys on its single
//!     mutex at this shape and loses 2x, the sharded one tracks fresh
//!     allocation. A lease still takes a lock where the TLS pool touches
//!     only thread-local storage — the single-thread gap above, worth a few
//!     ns/task over the `map_init` floor on tiny tasks at high worker
//!     counts.
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
            let mut buf = pool.get(); // shard lock/pop, then lock/push on drop
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
                    .map_init(|| vec![0.0; n], |buf, task| black_box(work(buf, task)))
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

    // Tiny parallel tasks: ~µs of work each — small enough that per-lease
    // lock traffic across 16 workers would show if it contended (the shard
    // locks are per-thread, and below the strategies tie).
    parallel(8192, 50_000, 1);
    // Huge parallel tasks: allocation dominates; all reuse strategies tie.
    parallel(262_144, 512, 2);
}
