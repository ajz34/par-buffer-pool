//! std-only: scoped threads, a non-`'static` initializer, and cloned handles.
//!
//! No rayon here — the same pool serves plain [`std::thread::scope`]
//! workers. This example exists to show the `'a` in
//! [`BufferPool<'a, T>`](par_buffer_pool::BufferPool) earning its keep:
//!
//!   * the initializer closures borrow fields of a plain stack-local
//!     `WorkspacePlan` — not `'static`, not `Clone`, no leaking;
//!   * each worker receives its *own handle* via [`BufferPool::clone`]
//!     (cheap, like cloning an `Arc`), so workers can even outlive the
//!     original binding as long as the scope lives;
//!   * the out pool registers a reset hook, so every lease starts zeroed
//!     without any `fill(0.0)` at the call site.
//!
//! The pool (and the plan it borrows from) die together at the end of `main`.
//!
//! Run with `cargo run --release --example scoped_threads`.

use std::thread;

use par_buffer_pool::BufferPool;

/// Per-call workspace sizing, borrowed (not moved, not cloned) by the pools.
struct WorkspacePlan {
    points: usize,
    nvar: usize,
    // imagine: device handles, operator tables, anything non-'static
}

fn main() {
    let plan = WorkspacePlan {
        points: 10_000,
        nvar: 4,
    };
    let nthreads = 8;
    let leases_per_thread = 500;

    // Both initializers borrow `plan` — possible because `new` bounds its
    // closure by `'a`, not `'static`.
    let scr_pool: BufferPool<Vec<f64>> = BufferPool::new(|| vec![0.0; plan.points]);
    let out_pool: BufferPool<Vec<f64>> =
        BufferPool::new(|| vec![0.0; plan.nvar]).with_reset(|buf| buf.fill(0.0));

    thread::scope(|s| {
        for t in 0..nthreads {
            // Clone handles into each worker (shared state, not shared buffers).
            let scr_pool = scr_pool.clone();
            let out_pool = out_pool.clone();
            s.spawn(move || {
                let mut grand_total = 0.0f64;
                for k in 0..leases_per_thread {
                    let mut scr = scr_pool.get();
                    // deterministic fill standing in for real evaluation
                    for (i, x) in scr.iter_mut().enumerate() {
                        *x = ((i * (t + 1) + k) % 100) as f64;
                    }

                    let mut out = out_pool.get(); // starts zeroed via with_reset
                    for v in 0..plan.nvar {
                        out[v] += scr.iter().sum::<f64>() * (v + 1) as f64;
                    }
                    grand_total += out[0];
                    // both guards return their buffers here — every iteration
                }
                println!("worker {t}: partial {grand_total:.3}");
            });
        }
    });

    // `plan` is still borrowed by the pools, so it must outlive them — and it
    // does; dropping everything in reverse order just works.
    let scr_stats = scr_pool.stats();
    let out_stats = out_pool.stats();
    println!(
        "scr: {} leases / {} allocations; out: {} leases / {} allocations",
        scr_stats.leases, scr_stats.allocations, out_stats.leases, out_stats.allocations
    );
    assert_eq!(scr_stats.leases, nthreads * leases_per_thread);
    assert!(scr_stats.allocations <= nthreads);
}
