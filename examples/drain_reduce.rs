//! `drain` as a parallel reduction: pooled accumulators instead of rayon's
//! `fold`/`reduce`.
//!
//! Summing a big vector is the canonical reduction, and the manual rayon
//! answer is `fold` + `sum`: an identity closure, a combine closure, and an
//! adapter chain to keep straight. The pool's answer is one `BufferPool`
//! whose buffers *are* the accumulators. Every parallel item adds into some
//! pooled accumulator — which one does not matter, because each
//! pop → add → push round trip preserves the running total — and once the
//! phase ends, `drain` is the combine step: it hands back every partial
//! sum, and an ordinary `.sum()` finishes the job.
//!
//! Where `fold`'s accumulators live inside one iterator chain, pooled ones
//! persist across loops and phases, work for any buffer type (a `Vec`, a
//! matrix, a histogram — anything `+=`-shaped), and stay inspectable: the
//! drained `Vec` is the list of partials, in parking order.
//!
//! Run with `cargo run --release --example drain_reduce`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

fn main() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let expected: f64 = data.iter().sum();

    // The initializer plays the role of `fold`'s identity element.
    let pool = BufferPool::new(|| 0.0f64);

    // The map phase: every item adds into a pooled accumulator. The guard is
    // a temporary of the statement, so the buffer parks again at the `;` —
    // one pop → add → push round trip per item. (Kept per-item to stay
    // minimal; real workloads lease once per chunk of work, amortizing the
    // pop/push the way the crate docs' lease-granularity advice suggests.)
    data.par_iter().for_each(|&x| *pool.get() += x);

    // The reduce phase: no guards are out at a phase boundary, so the drain
    // is exact — it hands back every partial sum the workers left behind.
    let partials = pool.drain();
    let total: f64 = partials.iter().sum();

    println!("sequential sum:        {expected}");
    println!(
        "pooled reduction:      {total} from {} partial sums",
        partials.len()
    );
    assert_eq!(total, expected);

    // How many partial sums ever existed is scheduling-dependent — rayon
    // assigns jobs, not resources, to threads — and typically lands near
    // the worker count, about what a `fold` would have built. The
    // deterministic facts are the ones worth asserting: every lease was
    // served, work actually happened, and every accumulator ever allocated
    // came back in the drain.
    let stats = pool.stats();
    println!(
        "{leases} leases, {alloc} accumulators ever allocated (nthreads = {nthreads})",
        leases = stats.leases,
        alloc = stats.allocations,
        nthreads = rayon::current_num_threads(),
    );
    assert_eq!(stats.leases, data.len());
    assert!(stats.allocations >= 1, "work actually happened");
    assert_eq!(
        partials.len(),
        stats.allocations,
        "nothing was capped or detached, so the drain is the whole accumulator set"
    );
}
