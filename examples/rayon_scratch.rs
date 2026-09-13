//! Canonical pattern: a scratch pool feeding a rayon loop.
//!
//! A batch of binary blobs (network payloads, cache entries, file chunks —
//! anything) is hex-encoded in parallel. Every task needs a temporary
//! `String` of the same shape, so a plain allocator would hand out one fresh
//! allocation per task; the pool allocates about one per worker thread and
//! then recycles it for the rest of the run.
//!
//! Run with `cargo run --release --example rayon_scratch`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const BLOBS: usize = 512;
const BYTES_PER_BLOB: usize = 4096;
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

fn main() {
    // Deterministic stand-ins for the input data.
    let blobs: Vec<Vec<u8>> = (0..BLOBS)
        .map(|b| (0..BYTES_PER_BLOB).map(|i| (b * 31 + i) as u8).collect())
        .collect();

    // One pool per scratch kind, built once *before* the parallel loop. The
    // closure captures nothing here; in real code it may borrow runtime
    // configuration from the caller — `new` never required 'static.
    let hex_pool = BufferPool::new(|| String::with_capacity(2 * BYTES_PER_BLOB));

    let total_len: usize = blobs
        .par_iter()
        .map(|blob| {
            // Lease a buffer. A guard, not a raw value: an early return or a
            // panic in the task body below cannot lose it.
            let mut hex = hex_pool.get();

            // Leased buffers keep whatever the last task left in them, so
            // start from a clean slate (or use `.with_reset` on the pool).
            hex.clear();
            for &byte in blob {
                hex.push(HEX_DIGITS[(byte >> 4) as usize] as char);
                hex.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
            }

            hex.len()
            // `hex` returns to the pool here — on success, early return, and
            // panic alike; there is no `put` call to keep in sync.
        })
        .sum();

    println!("encoded {BLOBS} blobs into {total_len} hex chars");

    // Peak scratch is bounded by ~nthreads buffers, not ntasks.
    let nthreads = rayon::current_num_threads();
    let stats = hex_pool.stats();
    println!(
        "{leases} leases, {alloc} allocations (nthreads = {nthreads}), {reuse} served from recycle",
        leases = stats.leases,
        alloc = stats.allocations,
        reuse = stats.reuses(),
    );
    assert_eq!(stats.leases, BLOBS);
    assert!(stats.allocations <= nthreads);
}
