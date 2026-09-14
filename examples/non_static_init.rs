//! Non-`'static` initializers: the pool borrows from the caller's frame.
//!
//! Most buffer pools demand `Fn() -> T + Send + Sync + 'static` — often
//! plus a `T: Clone` bound, or an `Arc` around everything the closure
//! captures — so per-run parameters have to be cloned or moved into the
//! initializer. [`BufferPool`]'s lifetime parameter `'a` is the whole
//! feature: the initializer may borrow plain stack locals, for exactly as
//! long as the pool lives. No `move`, no `Clone`, no leaking, and the
//! borrowed data stays owned (and usable) by its owner.
//!
//! The scenario: an audio-style equalizer. A per-band gain preset is a
//! stack local of `main`; the pool's initializer builds each fresh channel
//! buffer by copying the *borrowed* preset, and a reset hook (same `'a`)
//! restores the preset on every return, so every lease starts from the
//! same known state. Tasks apply a frame-local boost on top and report a
//! loudness; `main` then re-derives the expected loudness serially — which
//! is only possible because `preset` was never given away.
//!
//! (The lock-free [`ThreadLocalPool`](par_buffer_pool::ThreadLocalPool)
//! cannot offer this: its slots are process-global, so its initializer
//! must be `move` + `'static`.)
//!
//! Run with `cargo run --release --example non_static_init`.

use par_buffer_pool::BufferPool;
use rayon::prelude::*;

/// Builds one channel buffer from a *borrowed* preset.
///
/// The signature is the point: a plain `&[f64]` with an elided lifetime,
/// not an owned preset, not a `'static` table. Every fresh buffer starts
/// as a copy of whatever frame the pool's initializer borrowed.
fn channel_from_preset(preset: &[f64]) -> Vec<f64> {
    preset.to_vec() // copies the borrowed data; the copy is owned, the source is not given up
}

/// The frame-local boost applied to one band (deterministic).
fn boost(frame: usize) -> f64 {
    1.0 + (frame % 4) as f64 * 0.25 // 1.0, 1.25, 1.5, 1.75
}

fn main() {
    // A plain stack local: not 'static, and never cloned or moved anywhere.
    let preset = vec![0.8, 1.0, 1.2, 0.9, 1.1, 0.7, 1.3, 0.95];
    let bands = preset.len();
    let frames = 512;

    // The initializer captures `&preset` — a borrow of a local — which is
    // exactly what `'a` (instead of `'static`) allows. The reset hook
    // captures the same borrow, so even a recycled buffer starts at the
    // preset; without it, leases would inherit the previous frame's boost.
    let pool = BufferPool::new(|| channel_from_preset(&preset))
        .with_reset(|channel| channel.copy_from_slice(&preset));

    // Per frame: lease a channel at the preset, boost one band, measure.
    let loudness: Vec<f64> = (0..frames)
        .into_par_iter()
        .map(|frame| {
            let mut channel = pool.get(); // starts at the borrowed preset
            channel[frame % bands] *= boost(frame);
            channel.iter().sum::<f64>()
            // `channel` returns here and is re-seeded by the reset hook
        })
        .collect();

    // `preset` was never moved or cloned into the pool, so `main` still
    // owns it — this serial re-check *is* the feature demonstration. The
    // per-frame arithmetic is identical to the parallel path, so the
    // comparison is exact.
    for (frame, &measured) in loudness.iter().enumerate() {
        let mut expected = preset.clone(); // a scratch copy here, not a requirement of the pool
        expected[frame % bands] *= boost(frame);
        assert_eq!(expected.iter().sum::<f64>(), measured, "frame {frame}");
    }

    let stats = pool.stats();
    println!(
        "{frames} frames over {bands} bands, loudness range {:.3}..{:.3}",
        loudness.iter().cloned().fold(f64::MAX, f64::min),
        loudness.iter().cloned().fold(f64::MIN, f64::max),
    );
    println!(
        "{leases} leases, {alloc} allocations (nthreads = {nthreads}), {reuses} served from recycle",
        leases = stats.leases,
        alloc = stats.allocations,
        nthreads = rayon::current_num_threads(),
        reuses = stats.reuses(),
    );

    // Deterministic facts only: every frame got a lease, the pool worked,
    // and — scheduling-dependent as ever — fresh-buffer allocations stay
    // few because leases recycle.
    assert_eq!(stats.leases, frames);
    assert!(stats.allocations >= 1, "work actually happened");
    assert!(stats.allocations < frames, "recycling happened");
}
