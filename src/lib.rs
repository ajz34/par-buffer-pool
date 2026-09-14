//! # par-buffer-pool
//!
//! A tiny, dependency-free, `#![forbid(unsafe_code)]` crate with **two buffer
//! pools** with **RAII guards**, for reusing scratch buffers across parallel
//! workers (rayon, scoped threads, ...):
//!
//! - [`BufferPool`] — one shared, mutex-guarded pile; any thread can satisfy
//!   any lease. ([`SharedPooled`] guard, `prelude`.)
//! - [`ThreadLocalPool`] — one slot per worker thread; leases touch no lock
//!   and no shared cache line. ([`LocalPooled`] guard.)
//! - **Initializers that borrow.** [`BufferPool`]'s initializer — and its
//!   reset hook — are bounded by the pool's lifetime `'a`, not `'static`:
//!   capture plain stack locals (a dimensions tuple, a per-run preset) with
//!   no `move`, no `Clone`, no `Arc`, and the borrowed data stays owned by
//!   its owner. See
//!   [Non-`'static` initializers](#non-static-initializers-bufferpool-only)
//!   and the `non_static_init` example.
//!
//! Both exist because of a pattern that shows up in every parallel codebase:
//! a loop over many small work items where every item needs the same kind of
//! temporary buffer.
//!
//! ```text
//! (0..ntasks).into_par_iter().for_each(|task| {
//!     let mut scratch = vec![0.0; 1 << 20]; // <- fresh malloc per task
//!     let mut header = String::new();       // <- and another one
//!     // ... a few microseconds of work, then both are dropped ...
//! });
//! ```
//!
//! The buffers are large enough that allocating them costs more than the
//! work, and short-lived enough that they are immediately re-allocated by the
//! next task. Worse, peak memory is not *one* buffer but *nthreads* of them,
//! so the allocator gets hammered with big transient allocations from every
//! thread.
//!
//! Both pools fix that the same way — a lazily-filled pool plus a drop-based
//! lease:
//!
//! - `get` returns a guard, not a bare value; the guard derefs to the buffer.
//! - When the guard is dropped — normally, on an early return, or during a
//!   panic unwind — the buffer goes back automatically. **There is no `put`
//!   to forget.**
//! - The pool only allocates a new buffer when the relevant slot is
//!   momentarily empty, so the buffer count converges to the peak number of
//!   concurrently outstanding leases.
//!
//! ## Quick start
//!
//! ```rust
//! use par_buffer_pool::BufferPool;
//! use rayon::prelude::*;
//!
//! // One pool per scratch kind, created once *before* the parallel loop.
//! let n = 1024;
//! let pool = BufferPool::new(|| vec![0.0f64; n]);
//!
//! let sums: Vec<f64> = (0..64)
//!     .into_par_iter()
//!     .map(|task| {
//!         let mut buf = pool.get(); // fresh on first lease, recycled after
//!         buf.fill(task as f64);
//!         buf.iter().sum() // `buf` returns itself to the pool at scope end
//!     })
//!     .collect();
//!
//! let stats = pool.stats();
//! assert_eq!(stats.leases, 64); // every task took a lease
//! ```
//!
//! The same shape works verbatim with [`std::thread::scope`], async runtimes,
//! or a single thread — rayon is not required (it is only used in the
//! examples).
//!
//! ## Which pool?
//!
//! Both pools share one API shape — `new` / `get` (a guard) / `with` /
//! `put` / `into_inner` / `with_reset` / `stats`, plus `with_max_idle` and
//! `drain` on [`BufferPool`] only — so switching between them is mostly a
//! type swap. They differ in mechanism, and the mechanism shows
//! up in exactly one place: **many small tasks at high worker counts favor
//! [`ThreadLocalPool`]; everything else is a feature choice.**
//!
//! | | [`BufferPool`] | [`ThreadLocalPool`] |
//! |---|---|---|
//! | Storage | one shared `Mutex<Vec<T>>` | one slot per worker thread (`thread_local!`) |
//! | Lease cost | mutex lock/unlock pair (~26 ns, contended under many tiny tasks) | TLS access, no lock (~9 ns, never contended) |
//! | Initializer | may borrow non-`'static` data (`BufferPool<'a, T>`) | must be `'static` |
//! | Buffers per pool | ≈ peak concurrent leases | ≈ worker threads, even if only two are ever busy |
//! | Buffer movement | returns to the shared pile from any thread | parks on the thread that drops the guard; migrates if that is not where it was leased |
//! | After the pool is dropped | buffers freed with the pool | reclaimed on each thread's next pool interaction, or at thread exit (never unbounded) |
//! | Idle cap | `with_max_idle` | not needed: each thread parks at most one buffer |
//! | `idle_len` | global parked count | this thread's parked count (0 or 1) |
//! | `drain` | the whole idle pile at once (call it between phases: leases still out are silently not included) | none: other threads' slots are unreachable (pool drop / thread exit reclaims) |
//!
//! (Lease costs measured on a 16-core desktop CPU with glibc, default
//! features; see `local_static_bench` under [Examples](#examples) for the
//! full matrix and the caveat that orderings should be re-measured on
//! target hardware.)
//!
//! Practical guidance:
//!
//! - Tasks in the tens-of-µs range or larger: the lease mechanism vanishes
//!   into noise — pick by features. Need non-`'static` initializers, a
//!   global idle cap, `drain`, or buffers detached on one thread and
//!   recycled on another? [`BufferPool`]. Want each worker to keep its own scratch
//!   forever with zero shared state? [`ThreadLocalPool`].
//! - Many tasks in the single-digit µs range at many workers: prefer
//!   [`ThreadLocalPool`] — a shared mutex hit by millions of lease-pairs per
//!   second falls into futex-convoy territory and can end up *slower than
//!   not pooling at all* (see `local_static_bench` in the examples).
//! - Mixed workloads can simply use both: they are independent types over
//!   the same guard pattern.
//!
//! ## API tour
//!
//! | Item | What it does |
//! |---|---|
//! | [`new`](BufferPool::new) | Build a pool; `init` runs lazily, only when a lease finds the slot/pile empty. |
//! | [`get`](BufferPool::get) | Lease a buffer as a guard ([`SharedPooled`] / [`LocalPooled`], [`std::ops::Deref`] / [`std::ops::DerefMut`] to `T`). |
//! | guard drop | Return the buffer — on scope exit, early return, or panic unwind. |
//! | [`with`](BufferPool::with) | Closure-based checkout of the same mechanism. |
//! | `into_inner` | Detach the buffer when it *is* the result (not recycled). |
//! | [`put`](BufferPool::put) | Manual return for detached/raw buffers. |
//! | [`drain`](BufferPool::drain) | Empty the idle pile into a [`Vec`] — best effort: leases still out are silently not included and park again afterwards, so call it between phases, once every guard has been dropped. Reset hooks do not run; the pool keeps working. ([`ThreadLocalPool`] has none: per-thread slots are unreachable cross-thread.) |
//! | [`with_reset`](BufferPool::with_reset) | Run `f(&mut buf)` on every return, so leases start in a known state (e.g. zeroed). |
//! | [`with_max_idle`](BufferPool::with_max_idle) | Cap idle buffers; returns beyond the cap are dropped. ([`ThreadLocalPool`] needs no cap.) |
//! | [`stats`](BufferPool::stats) | `leases` / `allocations` counters on both pools — proof the pool works, and the number for memory budgeting. Behind the `stats` feature (off by default). |
//!
//! The pools never clear buffers on their own — accumulation buffers should
//! use [`with_reset`](BufferPool::with_reset), buffers fully overwritten by
//! every task should skip it.
//!
//! `BufferPool<'a, T>` is `Send + Sync` whenever `T: Send`; [`SharedPooled`]
//! is `Send` under the same condition, so leases may even migrate between
//! threads. [`ThreadLocalPool`] is always `Send + Sync`; its guard is `Send`
//! when `T: Send`.
//!
//! ## Examples
//!
//! The repository's
//! [`examples/` directory](https://github.com/ajz34/par-buffer-pool/tree/main/examples)
//! is the documentation of record; each file is runnable
//! (`cargo run --release --example <name>`), and the docs.rs build scrapes
//! their call sites into the per-item docs of the API they use.
//!
//! | Example | Pattern |
//! |---|---|
//! | `rayon_scratch` | The canonical one: a scratch pool feeding a rayon loop (hex-encoding binary blobs), stats printout. |
//! | `drain_reduce` | Reduction without `fold`: pooled `f64` accumulators collect partial sums in a plain parallel `for_each`; `drain` hands them back for the final combine. |
//! | `non_static_init` | The `'a` feature: initializer and reset hook borrow plain stack locals (an EQ preset copied into every fresh buffer) — no `'static`, no `move`, no `Clone`, and the borrowed data stays owned by its owner. |
//! | `pair_scores` | All-pairs tasks, one `O(n²)` scratch matrix per pair; shows churn drop from `O(ntasks)` buffers to a near-constant pooled count. |
//! | `detach_collect` | Scratch vs. result in the same task: Mandelbrot strips detach via `into_inner` into the image, escape-time scratch recycles. |
//! | `size_buckets` | Variable-size workloads: a grow-only pool (`clear` + `resize` per lease) and power-of-two size-class pools. |
//! | `scoped_threads` | No rayon: [`std::thread::scope`], cloned handles, an initializer borrowing stack-local config, and a reset hook. |
//! | `alloc_bench` | `fresh` vs `pooled` vs rayon `map_init` on 2 MiB buffers: wall time (best-of-N) plus allocation counts. |
//! | `local_static_bench` | The two pools head-to-head against `fresh`/`map_init`/a hoisted floor: lease-cost microbench plus parallel tiny/huge task regimes. |
//!
//! ## The guard is the point
//!
//! A manual `get`/`put` pool asks the caller to match every checkout with a
//! return, in every branch, on every error path, through every `?` and panic.
//! In practice the `put` calls end up duplicated per branch and are still
//! forgotten on some path. With a guard there is nothing to match:
//!
//! ```rust
//! # use par_buffer_pool::BufferPool;
//! fn process(pool: &BufferPool<Vec<f64>>, n: usize) -> f64 {
//!     let mut buf = pool.get();
//!     if n == 0 {
//!         return 0.0; // early return: buffer still goes back to the pool
//!     }
//!     buf.truncate(n);
//!     buf.iter().sum::<f64>() // and so does this one
//! }
//!
//! let pool = BufferPool::new(Vec::new);
//! assert_eq!(process(&pool, 0), 0.0); // early return: buffer recycled too
//! assert_eq!(process(&pool, 4), 0.0);
//! assert_eq!(pool.idle_len(), 1); // one buffer served both leases
//! ```
//!
//! [`ThreadLocalPool`] guards behave identically (including under panic
//! unwinds); when a named guard is awkward, both pools also offer a
//! closure-based checkout, `with(|buf| ...)`.
//!
//! ### Keeping a buffer: `into_inner`
//!
//! Sometimes the buffer is the *result* and must not be recycled. Detach it
//! with `into_inner` (`SharedPooled::into_inner` /
//! [`LocalPooled::into_inner`]), which moves the value out without returning
//! it to the pool:
//!
//! ```rust
//! # use par_buffer_pool::ThreadLocalPool;
//! # use rayon::prelude::*;
//! let pool = ThreadLocalPool::new(Vec::<u8>::new);
//!
//! let results: Vec<Vec<u8>> = (0..4)
//!     .into_par_iter()
//!     .map(|i| {
//!         let mut buf = pool.get();
//!         buf.extend_from_slice(&[i as u8; 3]);
//!         buf.into_inner() // keep it; this one will not be recycled
//!     })
//!     .collect();
//!
//! assert_eq!(results[3], vec![3, 3, 3]);
//! assert_eq!(pool.idle_len(), 0); // nothing was (or should be) parked
//! ```
//!
//! A raw value can also be recycled manually with `put`, e.g. a buffer that
//! was detached earlier and is now done serving as a result.
//!
//! ## The pools do not clear buffers
//!
//! Recycled buffers keep their previous contents by design — clearing is a
//! per-workload decision (accumulation buffers need it; buffers that get
//! fully overwritten anyway must not pay for it). If all leases of a pool
//! start from the same known state, register a reset hook with `with_reset`
//! (on either pool); it runs on every return, so every lease starts reset:
//!
//! ```rust
//! # use par_buffer_pool::BufferPool;
//! let pool = BufferPool::new(|| vec![0.0f64; 8]).with_reset(|buf| buf.fill(0.0));
//! {
//!     let mut buf = pool.get();
//!     buf[2] = 7.0;
//! } // dropped -> reset -> parked in the pool
//! assert!(pool.get().iter().all(|&x| x == 0.0)); // leased clean
//! ```
//!
//! ## Bounding idle memory
//!
//! [`BufferPool`] idle buffers are never freed while the pool lives, so
//! `with_max_idle` caps how many are kept. [`ThreadLocalPool`] needs no cap:
//! each thread parks at most one buffer per pool by construction, and
//! dropped pools have their parked buffers reclaimed (see the
//! [choosing guide](#which-pool)). To take parked buffers back out
//! explicitly, [`BufferPool`]'s `drain` moves the whole idle pile into a
//! `Vec`, leaving the pool empty but fully usable. It is best-effort by
//! design: leases still out are silently not included and park again after
//! the drain returns, so call it between phases, once every guard has been
//! dropped. [`ThreadLocalPool`] has no `drain` — its parked buffers sit in
//! per-thread slots no other thread can reach, and are reclaimed when the
//! pool is dropped or the parking thread exits.
//!
//! ## Non-`'static` initializers (`BufferPool` only)
//!
//! [`BufferPool`] carries a lifetime `'a` bounding its initializer, so the
//! closure may borrow from the caller — a dimensions tuple, a formatting
//! config, a connection handle — without requiring `'static` or cloning.
//! [`ThreadLocalPool`] initializers must be `'static` (a thread-local slot
//! cannot borrow from a caller's frame); `move` the data in instead.
//!
//! ```rust
//! # use par_buffer_pool::BufferPool;
//! use std::thread;
//!
//! let dims = [4_usize, 4]; // plain stack local, not 'static, not Clone
//! let pool = BufferPool::new(|| vec![0.0f64; dims[0] * dims[1]]);
//!
//! thread::scope(|s| {
//!     for t in 0..4 {
//!         let pool = &pool;
//!         s.spawn(move || {
//!             let mut buf = pool.get();
//!             buf[0] = t as f64;
//!         });
//!     }
//! }); // guards returned; pool dies with `dims`, no 'static bound anywhere
//! ```
//!
//! The `non_static_init` example builds a full run around this: initializer
//! and reset hook both borrow one stack-local EQ preset, and the borrowed
//! data is still owned — and re-checked serially — by `main` after the
//! parallel phase.
//!
//! ## Any buffer type, and views over it
//!
//! Both pools store whatever `T` you build — `Vec<f64>`, `String`, `Vec<u8>`,
//! or your own struct. And because guards deref to `T`, per-task views over
//! the pooled storage (sub-slices, or wrappers from view-based libraries such
//! as `ndarray` or `bytes`) are a one-liner inside each task; the storage
//! returns to its slot when the guard drops, view or no view:
//!
//! ```rust
//! # use par_buffer_pool::BufferPool;
//! // One pool per scratch kind, created once *before* the parallel section.
//! let frame_pool = BufferPool::new(|| vec![0u8; 4 + 64]); // header + payload
//!
//! let mut frame = frame_pool.get();
//! let (header, payload) = frame.split_at_mut(4); // views over pooled storage
//! header.copy_from_slice(&64u32.to_be_bytes());
//! payload.fill(b'.');
//! assert_eq!(frame.len(), 68);
//! // `frame` returns to the pool here; the views simply die with it
//! ```
//!
//! ## When *not* to use this
//!
//! - Buffers are small and cheap (`Vec<f64>` of a few KiB or less): the
//!   allocator is already a pool; prefer plain `vec![]` until measured.
//! - The parallel loop is a single `for_each` and buffers never outlive it:
//!   rayon's `for_each_init(|| ..., |scratch, item| ...)` keeps one scratch
//!   set per worker with no locking at all — and
//!   [`ThreadLocalPool::with`] is the same idea generalized beyond a single
//!   loop (one slot per thread per pool, reused across loops, with reset,
//!   detach, and stats).
//! - Raw `thread_local!` + `RefCell` scratch (the classic hand-rolled
//!   version) is famously awkward across library boundaries: no guard, no
//!   reset, no detach, values leaked on non-`'static` data, reentrancy
//!   panics. [`ThreadLocalPool`] keeps the access cost and adds the missing
//!   structure.
//!
//! ## Compared with other crates
//!
//! The [`comparison`] module surveys crates.io's buffer- and object-pool
//! crates — object-pool, opool, Cloudflare's buffer-pool, swimmer,
//! lifeguard, lockfree-object-pool, syncpool, and the adjacent arenas,
//! `bytes`, and connection pools — with an honest table of what each offers,
//! what none of them offer, and where this crate differs.
//!
//! ## Feature flags
//!
//! - **`stats`** (off by default): per-pool lease/allocation counters,
//!   exposed through `stats()` — `leases`, `allocations`, and `reuses()`.
//!   Costs one relaxed atomic add per lease on a *shared* cache line (for
//!   both pools); enable it to verify that recycling is happening or to
//!   budget scratch memory (`allocations * len * size_of::<T>()` bytes).
//!   The crate's tests and examples enable it automatically via a
//!   dev-dependency on the crate itself.
//!
//! ## Design notes
//!
//! - **`BufferPool` locking.** One [`std::sync::Mutex`] guards a `Vec<T>`;
//!   the critical section is a `pop`/`push` (tens of nanoseconds) while
//!   lease holders do microseconds-to-milliseconds of work, so contention
//!   is negligible next to the allocation it removes — unless tasks are
//!   tiny and workers numerous, which is [`ThreadLocalPool`]'s territory.
//!   Allocation of new buffers happens *outside* the lock. And even
//!   uncontended, a recycled shared-pool buffer tends to pay a cross-core
//!   cache-line transfer on its next use — a second point for
//!   [`ThreadLocalPool`] under heavy small-buffer churn.
//! - **`BufferPool` poisoning.** The lock is only ever held for a
//!   `pop`/`push`, so a poisoned mutex carries no damaged invariant; it is
//!   recovered from with [`std::sync::PoisonError::into_inner`] rather than
//!   panicking or leaking idle buffers.
//! - **`ThreadLocalPool` mechanics.** One `thread_local!` registry per
//!   element type maps dense pool ids to slots (`Vec<Option<T>>` shaped);
//!   leases are a TLS access plus an `Option::take`. When a pool is
//!   dropped it flips a shared liveness flag and bumps a global epoch;
//!   each thread's next registry access sweeps its own dead slots (one
//!   `Acquire` load on the fast path), and thread exit reclaims whatever
//!   was left. No `unsafe` anywhere.
//! - **Memory growth.** With the `stats` feature (see "Feature flags"
//!   above), `allocations` is an upper bound on concurrently outstanding
//!   leases, which is the number to multiply by buffer size when budgeting
//!   scratch memory (e.g. `allocations * len * size_of::<T>()` bytes for
//!   `Vec`-like buffers).
//! - **Clone handles.** Both pools are cheap handles around shared state
//!   (like `Arc`): clone them into worker closures or store them in a
//!   driver struct. `SharedPooled` guards keep their pool alive even if all
//!   handles are dropped; `LocalPooled` guards borrow the pool instead,
//!   which is what keeps their lease path free of reference counts.
//! - **No `unsafe`.** Neither pool uses `MaybeUninit`, pinning, or raw
//!   pointers; guards are plain owned values.
//!
//! ## Testing
//!
//! `cargo test` covers the guard semantics of both pools (scope exit, early
//! return, panic unwind, cross-thread drop/migration), detach/re-pool, reset
//! hooks, idle caps, `drain` (best-effort pile hand-back and its mid-phase
//! caveat), handle cloning, nested leases, reclamation of dropped pools'
//! parked buffers, non-`'static` initializers under [`std::thread::scope`],
//! and rayon stress tests driving 20 000 leases through both pools.
//!
//! ## Provenance
//!
//! Most of this crate — code, tests, examples, and documentation — was
//! written by AI coding agents (GLM-5.3 and GLM-5.3-flash) under human
//! direction; the [repository README](https://github.com/ajz34/par-buffer-pool)
//! carries the same note alongside licensing details.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod buffer;
mod local;

pub use crate::buffer::{BufferPool, PoolStats, SharedPooled};
pub use crate::local::{LocalPooled, ThreadLocalPool};

/// Comparison documentation, written against the crates.io landscape.
#[doc = include_str!("comparison.md")]
pub mod comparison {}

/// Convenience re-exports: both pools, their guards, and the stats type.
///
/// ```
/// use par_buffer_pool::prelude::*;
///
/// let shared = BufferPool::new(Vec::<u8>::new);
/// let local = ThreadLocalPool::new(Vec::<u8>::new);
/// let _a: SharedPooled<'_, Vec<u8>> = shared.get();
/// let _b: LocalPooled<'_, Vec<u8>> = local.get();
/// ```
pub mod prelude {
    pub use crate::buffer::{BufferPool, PoolStats, SharedPooled};
    pub use crate::local::{LocalPooled, ThreadLocalPool};
}
