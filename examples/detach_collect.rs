//! Scratch vs. results in one task: recycle the transient buffer, detach the
//! one that becomes the answer.
//!
//! A rayon job renders a Mandelbrot image in horizontal strips. Each task
//! leases two buffers:
//!
//!   * a `Vec<f64>` of escape-time values — transient scratch, returned to
//!     its pool automatically at scope end;
//!   * a `Vec<u8>` strip of pixels — the *result*, detached with
//!     [`SharedPooled::into_inner`](par_buffer_pool::SharedPooled::into_inner) so it is
//!     moved, contents and all, straight into the image. No copy, no recycle.
//!
//! The stats make the split visible: the scratch pool shows heavy reuse, the
//! strip pool shows every lease as an allocation — exactly right, since
//! those buffers are still alive as image rows.
//!
//! Run with `cargo run --release --example detach_collect`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const WIDTH: usize = 320;
const HEIGHT: usize = 160;
// More strips than worker threads, so the scratch pool demonstrably recycles
// (with only ~nthreads strips every lease would be concurrent and fresh).
const STRIPS: usize = 32;
const MAX_ITER: u32 = 100;

fn escape_time(cx: f64, cy: f64) -> f64 {
    let (mut x, mut y) = (0.0, 0.0);
    let mut iter = 0;
    while x * x + y * y <= 16.0 && iter < MAX_ITER {
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        iter += 1;
    }
    if iter >= MAX_ITER {
        0.0 // interior: black
    } else {
        (iter as f64 + 1.0 - (x * x + y * y).sqrt().log2().max(0.0)) / MAX_ITER as f64
    }
}

fn main() {
    let rows_per_strip = HEIGHT / STRIPS;

    // Scratch: escape-time values, reused across strips.
    let escape_pool = BufferPool::new(|| vec![0.0f64; WIDTH]);
    // Result strips: 8-bit pixels. Buffers from this pool are *not* expected
    // to come back — they become rows of the final image.
    let strip_pool = BufferPool::new(|| vec![0u8; WIDTH * rows_per_strip]);

    let image: Vec<Vec<u8>> = (0..STRIPS)
        .into_par_iter()
        .map(|strip| {
            let mut escape = escape_pool.get(); // returned at scope end
            let mut pixels = strip_pool.get(); // detached below

            for r in 0..rows_per_strip {
                let y = ((strip * rows_per_strip + r) as f64 / HEIGHT as f64 - 0.5) * 2.4;
                for (px, slot) in escape.iter_mut().enumerate() {
                    let x = (px as f64 / WIDTH as f64 - 0.5) * 3.5;
                    *slot = escape_time(x, y);
                }
                let row = &mut pixels[r * WIDTH..(r + 1) * WIDTH];
                for (byte, &e) in row.iter_mut().zip(escape.iter()) {
                    *byte = if e == 0.0 { 0 } else { (e * 255.0) as u8 };
                }
            }

            pixels.into_inner() // keep this one; `escape` still recycles
        })
        .collect();

    let checksum: u64 = image
        .iter()
        .map(|s| s.iter().map(|&b| b as u64).sum::<u64>())
        .sum();
    println!(
        "rendered {} strips into a {}x{} image, checksum {checksum}",
        image.len(),
        WIDTH,
        HEIGHT
    );

    let scratch_stats = escape_pool.stats();
    println!(
        "escape (scratch): {} leases, {} allocations, {} reuses",
        scratch_stats.leases,
        scratch_stats.allocations,
        scratch_stats.reuses()
    );
    let result_stats = strip_pool.stats();
    println!(
        "strip (result): {} leases, {} allocations, {} reuses",
        result_stats.leases,
        result_stats.allocations,
        result_stats.reuses()
    );

    // The scratch pool recycled; the result pool did not (and should not).
    assert!(escape_pool.stats().reuses() > 0);
    assert_eq!(strip_pool.stats().allocations, STRIPS);
}
