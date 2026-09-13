//! All-pairs pattern: one big scratch matrix per pair task.
//!
//! An interaction score over `n` items, each with a `k`-dimensional feature
//! vector (think recommendation or ML feature crossing): the loop over the
//! `n * (n - 1) / 2` item pairs is parallel, and each task fills a `k x k`
//! scratch matrix with the outer product
//!
//! ```text
//! g[p, q] = features_i[p] * features_j[q]
//! ```
//!
//! then folds it into a scalar score. The scratch is `O(k^2)` and there are
//! `O(n^2)` tasks: allocating naively churns `n^2 / 2 * k^2 * 8` bytes
//! through the allocator, while the pool keeps roughly one buffer per worker
//! thread alive for the whole loop.
//!
//! Run with `cargo run --release --example pair_scores`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const N_ITEMS: usize = 48;
const N_FEATURES: usize = 96;

fn main() {
    // Deterministic stand-in for the feature vectors.
    let features = |item: usize, p: usize| {
        let x = (item * 131 + p * 9176) % 8191;
        (x as f64 / 8191.0 - 0.5) * 0.2
    };

    // The pool stores raw `Vec<f64>` storage; each task views it as a matrix.
    let gram_pool = BufferPool::new(|| vec![0.0f64; N_FEATURES * N_FEATURES]);

    let pairs: Vec<(usize, usize)> = (0..N_ITEMS)
        .flat_map(|i| (i + 1..N_ITEMS).map(move |j| (i, j)))
        .collect();
    let npairs = pairs.len();

    // `g` is a guard; the drop at scope end is the only "put" — no
    // early-exit bookkeeping, even if the body grows branches later.
    let scores: Vec<f64> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mut g = gram_pool.get(); // (k x k) scratch, recycled

            // g[p, q] = features_i[p] * features_j[q]  (outer product)
            for p in 0..N_FEATURES {
                let fi = features(i, p);
                for q in 0..N_FEATURES {
                    g[p * N_FEATURES + q] = fi * features(j, q);
                }
            }

            // Fold the matrix to a scalar: sum of squares.
            g.iter().map(|x| x * x).sum()
            // `g` returns to the pool here — including on the panic path
        })
        .collect();

    let total: f64 = scores.iter().sum();
    println!("total interaction score: {total:.6}");

    let stats = gram_pool.stats();
    let buffer_mb = (N_FEATURES * N_FEATURES * 8) as f64 / 1e6;
    let naive_churn = npairs as f64 * buffer_mb;
    let pooled = stats.allocations as f64 * buffer_mb;
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
