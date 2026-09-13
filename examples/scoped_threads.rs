//! std-only: scoped threads, a non-`'static` initializer, and cloned handles.
//!
//! No rayon here — the same pool serves plain [`std::thread::scope`]
//! workers. This example exists to show the `'a` in
//! [`BufferPool<'a, T>`](par_buffer_pool::BufferPool) earning its keep:
//!
//!   * the initializer borrows a field of a plain stack-local `Settings` —
//!     not `'static`, not `Clone`, no leaking;
//!   * each worker receives its *own handle* via [`BufferPool::clone`]
//!     (cheap, like cloning an `Arc`);
//!   * the pool registers a reset hook, so every lease starts empty without
//!     any `clear()` at the call site.
//!
//! Run with `cargo run --release --example scoped_threads`.

use std::thread;

use par_buffer_pool::BufferPool;

/// Formatting settings, borrowed (not moved, not cloned) by the pool.
struct Settings {
    columns: usize,
    separator: char,
}

fn main() {
    let settings = Settings {
        columns: 16,
        separator: ';',
    };
    let nthreads = 4;
    let rows_per_thread = 250;

    // The initializer borrows `settings` — a plain stack local. This works
    // because `new` bounds its closure by `'a`, not `'static`. The reset hook
    // empties each row on its way back into the pool.
    let row_pool = BufferPool::new(|| String::with_capacity(settings.columns * 2))
        .with_reset(|row| row.clear());

    thread::scope(|s| {
        for t in 0..nthreads {
            // Clone a handle into each worker (shared state, not shared buffers).
            let row_pool = row_pool.clone();
            s.spawn(move || {
                for k in 0..rows_per_thread {
                    let mut row = row_pool.get(); // starts empty via the reset hook
                    for c in 0..settings.columns {
                        if c > 0 {
                            row.push(settings.separator);
                        }
                        row.push((b'0' + ((t + k) % 10) as u8) as char);
                    }
                    // ... queue or send `row` somewhere ...
                    assert_eq!(row.len(), 2 * settings.columns - 1);
                } // `row` returns to the pool here, every iteration
            });
        }
    });

    // `settings` is still borrowed by the pool, so it must outlive it — and
    // it does; dropping everything in reverse order just works.
    let stats = row_pool.stats();
    println!(
        "{} leases, {} allocations (nthreads = {nthreads}), {} served from recycle",
        stats.leases,
        stats.allocations,
        stats.reuses()
    );
    assert_eq!(stats.leases, nthreads * rows_per_thread);
    assert!(stats.allocations <= nthreads);
}
