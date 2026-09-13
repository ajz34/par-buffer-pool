//! Pair-task pattern: one scratch matrix per (i, j) orbital pair.
//!
//! Mirrors RI-MP2/PT2 pair-energy code (`pure_pt2_pair_eng.rs`): the loop
//! over `nocc * (nocc - 1) / 2` occupied-orbital pairs is parallel, and each
//! task contracts two slices of a 3-center integral tensor into an
//! `(nvir x nvir)` scratch matrix
//!
//! ```text
//! g_ij[p, q] = sum_a L[a, p, i] * L[a, q, j]
//! ```
//!
//! then folds it into a scalar pair energy. The scratch is `O(nvir^2)` and
//! there are `O(nocc^2)` tasks: allocating naively churns
//! `nocc^2 / 2 * nvir^2 * 8` bytes through the allocator, while the pool
//! keeps roughly one buffer per worker thread alive for the whole loop.
//!
//! The same file also shows the *pre-pool* shape of this code in comments:
//! manual `get`/`put` with one `put` per early-exit branch.
//!
//! Run with `cargo run --release --example tiled_contraction`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const NOCC: usize = 32;
const NVIR: usize = 96;
const NAUX: usize = 48; // auxiliary RI basis size

fn main() {
    // Deterministic stand-ins: 3-center integrals L[a, p, i] and orbital
    // energies. Real code reads these from tensors; the pooling pattern is
    // identical either way.
    let l = |a: usize, p: usize, i: usize| {
        let x = (a * 9176 + p * 131 + i * 31) % 8191;
        (x as f64 / 8191.0 - 0.5) * 0.2
    };
    let eps_o = |i: usize| -10.0 + i as f64 * 0.3;
    let eps_v = |p: usize| 1.0 + p as f64 * 0.15;

    // The pool stores the *storage*; tasks view it as a matrix. Here the
    // initializer is a plain zeroed Vec — in rstsr code it would be
    // `rt::zeros(([NVIR, NVIR].f(), &device))` (an owned tensor), borrowed
    // into the pool's closure via the `'a` lifetime.
    let scr_pool = BufferPool::new(|| vec![0.0f64; NVIR * NVIR]);

    let pairs: Vec<(usize, usize)> = (0..NOCC)
        .flat_map(|i| (i + 1..NOCC).map(move |j| (i, j)))
        .collect();
    let npairs = pairs.len();

    // Pre-pool, this loop looked like:
    //
    //     let mut g = scr_pool.get();          // manual checkout ...
    //     ... three different early exits? three `put` calls to keep straight
    //     scr_pool.put(g);                     // ... and one at the end
    //
    // Now `g` is a guard; the drop at scope end is the only "put".
    let pair_energies: Vec<f64> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mut g = scr_pool.get(); // (nvir x nvir) scratch, recycled

            // g[p, q] = sum_a L[a, p, i] * L[a, q, j]
            for p in 0..NVIR {
                for q in 0..NVIR {
                    let mut acc = 0.0;
                    for a in 0..NAUX {
                        acc += l(a, p, i) * l(a, q, j);
                    }
                    g[p * NVIR + q] = acc;
                }
            }

            // Fold to a scalar: g^2 over MP2-like denominators.
            let mut e = 0.0;
            for p in 0..NVIR {
                for q in 0..NVIR {
                    let denom = eps_v(p) + eps_v(q) - eps_o(i) - eps_o(j);
                    e += g[p * NVIR + q] * g[p * NVIR + q] / denom;
                }
            }
            e
            // `g` returns to the pool here — including on the panic path
        })
        .collect();

    let total: f64 = pair_energies.iter().sum();
    println!("correlation energy (arb. units): {total:.9}");

    let stats = scr_pool.stats();
    let naive_churn = npairs as f64 * (NVIR * NVIR * 8) as f64 / 1e6;
    let pooled = stats.allocations as f64 * (NVIR * NVIR * 8) as f64 / 1e6;
    println!(
        "{} pair tasks, {} leases, {} allocations (nthreads = {})",
        npairs,
        stats.leases,
        stats.allocations,
        rayon::current_num_threads()
    );
    println!("scratch churn: naive {naive_churn:.1} MB vs pooled {pooled:.2} MB");
    assert!(stats.allocations <= rayon::current_num_threads());
}
