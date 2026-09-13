//! Canonical pattern: scratch buffers in a rayon loop over DFT-style grid
//! chunks.
//!
//! This mirrors the numerical-integration drivers of rstsr-based
//! quantum-chemistry code (e.g. `pure_eval_rho.rs`): a large quadrature grid
//! is processed in chunks of `nchunk` points, and each chunk task needs
//!
//!   * a scratch buffer `[nchunk * nao]` for the AO-to-density contraction,
//!   * an output buffer `[nchunk * nvar]` of per-point properties,
//!
//! both accumulated into shared results under a lock at the end of the task.
//! A plain allocator hands out `2 * ntask` big buffers per call; the pool
//! allocates about `2 * nthreads` of them once, then recycles.
//!
//! Run with `cargo run --release --example rayon_scratch`.

use std::sync::Mutex;

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

// Chunked-grid parameters in the style of real drivers: chunks of a few
// hundred grid points, batched across the pool's worker threads.
const NCHUNK: usize = 384;
const NAO: usize = 64; // number of basis functions
const NVAR: usize = 5; // properties per grid point
const NTASK: usize = 64; // grid chunks to process

fn main() {
    // Deterministic stand-ins for the density matrix and quadrature weights.
    let dm: Vec<f64> = (0..NAO * NAO)
        .map(|k| (((k * 2654435761) % 1000) as f64 / 1000.0 - 0.5) * 0.1)
        .collect();
    let weights: Vec<f64> = (0..NTASK * NCHUNK).map(|g| 1.0 / (g + 1) as f64).collect();
    // Deterministic "AO values" ao(g, a) in [-1, 1].
    let ao = |g: usize, a: usize| {
        let x = (g * 374761393 + a * 668265263) % 1000003;
        (x as f64 / 500001.5) - 1.0
    };

    // One pool per scratch kind, built once before the loop. The closures
    // capture `NCHUNK`/`NAO`-style locals by reference only because they are
    // consts here; in real code they typically borrow runtime dimensions (and
    // a device handle) from the caller — `new` never required 'static.
    let scr_pool = BufferPool::new(|| vec![0.0f64; NCHUNK * NAO]);
    let out_pool = BufferPool::new(|| vec![0.0f64; NCHUNK * NVAR]);

    // Shared accumulation target. Real tensor code often locks a `Mutex<()>`
    // token and then mutates disjoint slices of a shared tensor; with plain
    // arrays the honest equivalent is just a `Mutex<Vec<f64>>`.
    let out_tot = Mutex::new(vec![0.0f64; NVAR]);

    (0..NTASK).into_par_iter().for_each(|itask| {
        // Lease both buffers. Guards, not raw values: early returns and
        // panics in the task body below cannot leak them.
        let mut scr_buf = scr_pool.get();
        let mut out_buf = out_pool.get();

        // The out buffer is accumulated into, so it must start at zero.
        // (Equivalently: build the pool with `.with_reset(|b| b.fill(0.0))`
        // and delete this line.)
        out_buf.fill(0.0);

        // scr[c, a] = sum_k ao(g_c, k) * dm[k, a]  ("AO values" times density;
        // in rstsr code this is `rt::asarray((&mut *scr_buf, ...))` followed
        // by `scr.matmul_from(&ao_chunk, &dm, 1.0, 0.0)` — note beta = 0
        // fully overwrites, so scratch needs no clearing).
        let base = itask * NCHUNK;
        for c in 0..NCHUNK {
            for a in 0..NAO {
                let mut acc = 0.0;
                for k in 0..NAO {
                    acc += ao(base + c, k) * dm[k * NAO + a];
                }
                scr_buf[c * NAO + a] = acc;
            }
        }

        // Per-point "properties": powers of the density at each grid point.
        for c in 0..NCHUNK {
            let rho: f64 = (0..NAO).map(|a| scr_buf[c * NAO + a]).sum();
            for v in 0..NVAR {
                out_buf[c * NVAR + v] += weights[base + c] * rho.powi(v as i32 + 1);
            }
        }

        // Reduce into the shared output under the lock, then let the guards
        // return the buffers on scope exit — no `put` calls to keep in sync
        // with every branch of the task body.
        {
            let mut out = out_tot.lock().unwrap();
            for v in 0..NVAR {
                out[v] += (0..NCHUNK).map(|c| out_buf[c * NVAR + v]).sum::<f64>();
            }
        }
    });

    let out = out_tot.into_inner().unwrap();
    println!("properties (arb. units): {:.6?}", out);

    // Peak scratch is bounded by nthreads buffers per pool, not ntask:
    //   nthreads * nchunk * (nao + nvar) * 8 bytes.
    let nthreads = rayon::current_num_threads();
    for (name, pool) in [("scr", &scr_pool), ("out", &out_pool)] {
        let stats = pool.stats();
        println!(
            "{name} pool: {leases} leases, {alloc} allocations (nthreads = {nthreads}), \
             {reuse} served from recycle",
            leases = stats.leases,
            alloc = stats.allocations,
            reuse = stats.reuses(),
        );
        assert!(stats.allocations <= nthreads);
    }
}
