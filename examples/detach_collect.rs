//! Scratch vs. results in one task: recycle the transient buffer, detach the
//! one that becomes the answer.
//!
//! A rayon job renders a Mandelbrot image in horizontal strips. Each task
//! uses two leased buffers:
//!
//!   * a `Vec<f64>` of smooth escape-time values — transient scratch, returned
//!     to its pool automatically at scope end;
//!   * a `Vec<u8>` strip of pixels — the *result*, detached with
//!     [`par_buffer_pool::Pooled::into_inner`] so it is moved, contents and
//!     all, straight into the image. No copy, no recycle.
//!
//! The stats make the split visible: the scratch pool shows heavy reuse, the
//! strip pool shows every lease as an allocation — which is exactly right,
//! since those buffers are still alive as the image rows.
//!
//! Run with `cargo run --release --example detach_collect`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 320;
// More strips than worker threads, so the scratch pool demonstrably recycles
// (with only ~nthreads strips every lease would be concurrent and fresh).
const STRIPS: usize = 64;
const MAX_ITER: u32 = 250;

fn smooth_escape(cx: f64, cy: f64) -> f64 {
    let (mut x, mut y) = (0.0, 0.0);
    let mut iter = 0;
    while x * x + y * y <= 16.0 && iter < MAX_ITER {
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        iter += 1;
    }
    if iter >= MAX_ITER {
        0.0
    } else {
        // smooth coloring value in (0, 1)
        let mag = (x * x + y * y).sqrt();
        (iter as f64 + 1.0 - mag.log2().max(0.0)) / MAX_ITER as f64
    }
}

fn main() {
    let rows_per_strip = HEIGHT / STRIPS;

    // Scratch: smooth escape values, reused across strips.
    let escape_pool = BufferPool::new(|| vec![0.0f64; WIDTH]);
    // Result strips: 8-bit pixels. Buffers from this pool are *not* expected
    // to come back — they become rows of the final image.
    let strip_pool = BufferPool::new(|| vec![0u8; WIDTH * rows_per_strip]);

    let image: Vec<Vec<u8>> = (0..STRIPS)
        .into_par_iter()
        .map(|strip| {
            // Transient scratch: every strip borrows a buffer from the pool
            // and gives it back at scope end.
            let mut escape = escape_pool.get();

            // Result storage: this buffer will outlive the task as part of
            // the image, so lease it through the pool for uniformity but
            // detach it below.
            let mut pixels = strip_pool.get();

            for r in 0..rows_per_strip {
                let y = ((strip * rows_per_strip + r) as f64 / HEIGHT as f64 - 0.5) * 2.4;
                for px in 0..WIDTH {
                    let x = (px as f64 / WIDTH as f64 - 0.5) * 3.5;
                    escape[px] = smooth_escape(x, y);
                }
                let row = &mut pixels[r * WIDTH..(r + 1) * WIDTH];
                for (byte, &e) in row.iter_mut().zip(escape.iter()) {
                    // simple palette: interior black, rim shaded
                    *byte = if e == 0.0 { 0 } else { (e * 255.0) as u8 };
                }
            }

            pixels.into_inner() // detach: keep the buffer as this strip
                                // (`escape` still returns itself, above)
        })
        .collect();

    let checksum: u64 = image
        .iter()
        .map(|s| s.iter().map(|&b| b as u64).sum::<u64>())
        .sum();
    println!(
        "rendered {} strips, {}x{} px, checksum {checksum}",
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
