//! # par-buffer-pool
//!
//! A tiny, dependency-free, thread-safe **buffer pool** with **RAII guards**,
//! for reusing scratch buffers across parallel workers (rayon, scoped threads,
//! ...).
//!
//! It exists because of a pattern that shows up in every parallel codebase:
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
//! The buffers are large enough that allocating them costs more than the work,
//! and short-lived enough that they are immediately re-allocated by the next
//! task. Worse, peak memory is not *one* buffer but *nthreads* of them, so the
//! allocator gets hammered with big transient allocations from every thread.
//!
//! `par-buffer-pool` fixes that with a lazily-filled pool plus a drop-based
//! lease:
//!
//! - [`BufferPool::get`] returns a [`Pooled`] guard, not a bare value.
//! - The guard derefs to the buffer, so it is used like a normal `T`.
//! - When the guard is dropped — normally, on an early return, or during a
//!   panic unwind — the buffer goes back to the pool automatically. **There is
//!   no `put` to forget.**
//! - The pool only allocates a new buffer when it is momentarily empty, so the
//!   number of allocated buffers converges to the peak number of concurrently
//!   outstanding leases (≈ the number of worker threads).
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
//! assert!(stats.allocations <= rayon::current_num_threads()); // ≈ one per worker
//! ```
//!
//! The same pool works verbatim with [`std::thread::scope`], async runtimes, or
//! a single thread — rayon is not required (it is only used in the examples).
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
//! ### Keeping a buffer: [`Pooled::into_inner`]
//!
//! Sometimes the buffer is the *result* and must not be recycled. Detach it
//! with [`Pooled::into_inner`], which moves the value out without returning it
//! to the pool:
//!
//! ```rust
//! # use par_buffer_pool::BufferPool;
//! # use rayon::prelude::*;
//! let pool = BufferPool::new(Vec::<u8>::new);
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
//! assert_eq!(pool.idle_len(), 0); // nothing was (or should be) returned
//! ```
//!
//! A raw value can also be recycled manually with [`BufferPool::put`], e.g. a
//! buffer that was detached earlier and is now done serving as a result.
//!
//! ## The pool does not clear buffers
//!
//! Recycled buffers keep their previous contents by design — clearing is a
//! per-workload decision (accumulation buffers need it; buffers that get fully
//! overwritten anyway must not pay for it). If all
//! leases of a pool start from the same known state, register a reset hook with
//! [`BufferPool::with_reset`]; it runs on every return, so every lease starts
//! reset:
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
//! Idle buffers are never freed while the pool lives. Under steady load the
//! pool naturally holds no more buffers than there are workers, but a burst of
//! concurrent leases (or one odd call) can leave a larger pile parked forever.
//! [`BufferPool::with_max_idle`] caps how many idle buffers are kept; returns
//! beyond the cap are dropped instead of parked.
//!
//! ## Non-`'static` initializers
//!
//! [`BufferPool`] carries a lifetime `'a` bounding its initializer, so the
//! closure may borrow from the caller — a dimensions tuple, a formatting
//! config, a connection handle — without requiring `'static` or cloning:
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
//! ## Any buffer type, and views over it
//!
//! The pool stores whatever `T` you build — `Vec<f64>`, `String`, `Vec<u8>`,
//! or your own struct. And because the guard derefs to `T`, per-task views
//! over the pooled storage (sub-slices, or wrappers from view-based libraries
//! such as `ndarray` or `bytes`) are a one-liner inside each task; the
//! storage returns to the pool when the guard drops, view or no view:
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
//!   rayon's `for_each_init(|| ..., |scratch, item| ...)` keeps one scratch set
//!   per worker with no locking at all. A [`BufferPool`] earns its keep when
//!   several loop kinds share buffers, buffers must detach into results, the
//!   code must also run under plain threads, or the pool is stored in a driver
//!   struct and reused across calls.
//! - `thread_local!` scratch was the classic alternative and is notoriously
//!   awkward for generic scratch (no destructuring across library boundaries,
//!   leaked on non-`'static` data, awkward to reset between phases); a shared
//!   pool sidesteps all of that.
//!
//! ## Feature flags
//!
//! - **`stats`** (off by default): per-pool lease/allocation counters,
//!   exposed through `stats()` — `leases`, `allocations`, and `reuses()`.
//!   Costs one relaxed atomic add per lease; enable it to verify that
//!   recycling is happening or to budget scratch memory
//!   (`allocations * len * size_of::<T>()` bytes). The crate's tests and
//!   examples enable it automatically via a dev-dependency on the crate
//!   itself.
//!
//! ## Design notes
//!
//! - **Locking.** One [`std::sync::Mutex`] guards a `Vec<T>`; the critical
//!   section is a `pop`/`push` (tens of nanoseconds) while lease holders do
//!   microseconds-to-milliseconds of work, so contention is negligible next to
//!   the allocation it removes. Allocation of new buffers happens *outside*
//!   the lock.
//! - **Poisoning.** The lock is only ever held for a `pop`/`push`, so a
//!   poisoned mutex carries no damaged invariant; it is recovered from with
//!   [`std::sync::PoisonError::into_inner`] rather than panicking or leaking
//!   idle buffers.
//! - **Memory growth.** With the `stats` feature (see "Feature flags"
//!   above), `allocations` is an upper bound on concurrently outstanding
//!   leases, which is the number to multiply by buffer size when budgeting
//!   scratch memory (e.g. `allocations * len * size_of::<T>()` bytes for
//!   `Vec`-like buffers).
//! - **`Clone` handles.** [`BufferPool`] is a cheap handle around shared
//!   state (like `Arc`): clone it into worker closures or store it in a driver
//!   struct; guards keep the pool alive even if all handles are dropped.
//! - **No `unsafe`.** [`Pooled`] is a plain owned guard — no pinning, no
//!   `MaybeUninit`, nothing to leak by construction.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::fmt;
use std::ops::{Deref, DerefMut};
#[cfg(feature = "stats")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Counters describing what a [`BufferPool`] has done so far.
///
/// See [`BufferPool::stats`]. The difference `leases - allocations` is the
/// number of leases served from recycled buffers.
#[cfg(feature = "stats")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Total number of leases taken via [`BufferPool::get`].
    pub leases: usize,
    /// Number of times the initializer had to run, i.e. buffers created from
    /// scratch because the pool was momentarily empty.
    ///
    /// This is an upper bound on the number of concurrently outstanding
    /// leases, so `allocations * buffer_size` bounds the scratch memory the
    /// pool can be responsible for.
    pub allocations: usize,
}

#[cfg(feature = "stats")]
impl PoolStats {
    /// Leases served from an idle (recycled) buffer: `leases - allocations`.
    pub fn reuses(&self) -> usize {
        self.leases - self.allocations
    }
}

/// A thread-safe pool of reusable buffers of type `T`.
///
/// Call [`get`](BufferPool::get) to lease a buffer as a [`Pooled`] guard; the
/// buffer returns itself to the pool when the guard is dropped. See the
/// [crate documentation](crate) for the full story, examples, and patterns.
///
/// `BufferPool` is a cheap shared-state handle: it is [`Clone`] (no `T: Clone`
/// required), and `Send + Sync` whenever `T: Send`, so it can be shared with
/// rayon or scoped threads by reference or by cloning.
///
/// The lifetime `'a` bounds the initializer closure, allowing it to borrow
/// non-`'static` data (dimensions, configuration, shared handles) from the
/// surroundings; the pool simply must not outlive what its initializer
/// borrows.
///
/// # Example
///
/// ```
/// # use par_buffer_pool::BufferPool;
/// // lazily allocates at most ~nthreads buffers of 4 KiB each
/// let pool = BufferPool::new(|| vec![0u8; 4096]);
/// let mut buf = pool.get();
/// assert_eq!(buf.len(), 4096);
/// buf[0] = 1;
/// drop(buf);
/// assert_eq!(pool.idle_len(), 1);
/// ```
pub struct BufferPool<'a, T> {
    inner: Arc<PoolInner<'a, T>>,
}

/// The boxed initializer stored inside a pool.
type InitFn<'a, T> = Box<dyn Fn() -> T + Send + Sync + 'a>;

/// The boxed reset hook stored inside a pool (when one is registered).
type ResetFn<'a, T> = Box<dyn Fn(&mut T) + Send + Sync + 'a>;

struct PoolInner<'a, T> {
    idle: Mutex<Vec<T>>,
    init: InitFn<'a, T>,
    reset: Option<ResetFn<'a, T>>,
    max_idle: usize,
    #[cfg(feature = "stats")]
    leases: AtomicUsize,
    #[cfg(feature = "stats")]
    allocations: AtomicUsize,
}

impl<'a, T> BufferPool<'a, T> {
    /// Creates a pool whose buffers are made by `init`.
    ///
    /// `init` runs lazily, only when a lease finds the pool empty — never
    /// eagerly. It may borrow from the surroundings (hence `'a` instead of a
    /// `'static` bound). The pool starts empty with no idle cap; see
    /// [`with_max_idle`](BufferPool::with_max_idle) and
    /// [`with_reset`](BufferPool::with_reset) for the optional knobs.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let n = 256;
    /// // the closure borrows the stack local `n`; no 'static or move needed
    /// let pool = BufferPool::new(|| vec![0.0f64; n]);
    /// assert_eq!(pool.get().len(), 256);
    /// ```
    pub fn new(init: impl Fn() -> T + Send + Sync + 'a) -> Self {
        BufferPool {
            inner: Arc::new(PoolInner {
                idle: Mutex::new(Vec::new()),
                init: Box::new(init),
                reset: None,
                max_idle: usize::MAX,
                #[cfg(feature = "stats")]
                leases: AtomicUsize::new(0),
                #[cfg(feature = "stats")]
                allocations: AtomicUsize::new(0),
            }),
        }
    }

    /// Caps the number of idle buffers kept in the pool (builder style).
    ///
    /// When the pool already holds `max_idle` buffers, further returns are
    /// dropped instead of parked. Any `max_idle` of at least the expected
    /// worker count keeps steady-state recycling intact; a cap exists so that
    /// a burst of concurrency cannot park its buffers in the pool forever.
    /// The default is unbounded.
    pub fn with_max_idle(mut self, max_idle: usize) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("freshly built pool is uniquely owned")
            .max_idle = max_idle;
        self
    }

    /// Registers `reset` to run each time a buffer is returned (builder
    /// style), so that every lease starts from a known state.
    ///
    /// `reset` runs when a [`Pooled`] guard is dropped and on manual
    /// [`put`](BufferPool::put) — i.e. on *every* path into the pool. Buffers
    /// discarded because [`with_max_idle`](BufferPool::with_max_idle) is
    /// reached are dropped without being reset.
    ///
    /// Typical use: `|buf: &mut Vec<f64>| buf.fill(0.0)` for accumulation
    /// buffers. Skip the hook entirely when every consumer fully overwrites
    /// the buffer anyway — resets are not free.
    pub fn with_reset(mut self, reset: impl Fn(&mut T) + Send + Sync + 'a) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("freshly built pool is uniquely owned")
            .reset = Some(Box::new(reset));
        self
    }

    /// Leases a buffer, returning a [`Pooled`] guard.
    ///
    /// Recycles an idle buffer when one is available, otherwise runs the
    /// initializer (outside any lock). The buffer goes back to the pool when
    /// the guard is dropped — keep the guard for as long as the buffer is in
    /// use, e.g. bind it for the body of the task closure.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(String::new);
    /// let mut greeting = pool.get();
    /// greeting.push_str("hello");
    /// assert_eq!(&*greeting, "hello");
    /// ```
    pub fn get(&self) -> Pooled<'a, T> {
        #[cfg(feature = "stats")]
        self.inner.leases.fetch_add(1, Ordering::Relaxed);
        // Pop under the lock, but run the initializer outside it: allocation
        // can take arbitrarily long and must not stall other leases.
        let buffer = self.inner.lock_idle().pop().unwrap_or_else(|| {
            #[cfg(feature = "stats")]
            self.inner.allocations.fetch_add(1, Ordering::Relaxed);
            (self.inner.init)()
        });
        Pooled {
            buffer: Some(buffer),
            inner: Arc::clone(&self.inner),
        }
    }

    /// Returns a raw buffer to the pool, dropping it if the pool is full.
    ///
    /// This is the manual escape hatch, for values that were detached with
    /// [`Pooled::into_inner`] earlier or produced independently; the normal
    /// path is simply dropping a [`Pooled`] guard. If a reset hook was
    /// registered with [`with_reset`](BufferPool::with_reset), it runs here
    /// too.
    pub fn put(&self, buffer: T) {
        self.inner.recycle(buffer);
    }

    /// Number of idle buffers currently parked in the pool.
    pub fn idle_len(&self) -> usize {
        self.inner.lock_idle().len()
    }

    /// Lease and allocation counters; see [`PoolStats`].
    ///
    /// Only available with the `stats` feature (off by default).
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(Vec::<u8>::new);
    /// for k in 0..10 {
    ///     let buf = pool.get();
    ///     drop(buf); // returned, so the next lease recycles it
    ///     let _ = k;
    /// }
    /// let stats = pool.stats();
    /// assert_eq!(stats.leases, 10);
    /// assert_eq!(stats.allocations, 1); // one buffer served all ten leases
    /// assert_eq!(stats.reuses(), 9);
    /// ```
    #[cfg(feature = "stats")]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            leases: self.inner.leases.load(Ordering::Relaxed),
            allocations: self.inner.allocations.load(Ordering::Relaxed),
        }
    }
}

impl<'a, T> PoolInner<'a, T> {
    /// Locks the idle stack, recovering from poisoning.
    ///
    /// The lock only ever guards a `pop`/`push` of plain values, so a panic
    /// elsewhere can leave it poisoned but never leaves damaged data behind;
    /// recovering beats panicking on every later lease and beats leaking the
    /// idle buffers.
    fn lock_idle(&self) -> MutexGuard<'_, Vec<T>> {
        self.idle.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Parks `buffer` in the pool (resetting it first, if a hook is set),
    /// dropping it when the idle cap is reached.
    fn recycle(&self, buffer: T) {
        let mut buffer = buffer;
        if let Some(reset) = &self.reset {
            reset(&mut buffer);
        }
        let mut idle = self.lock_idle();
        if idle.len() < self.max_idle {
            idle.push(buffer);
        }
        // else: pool full — `buffer` drops here, which is the point of the cap
    }
}

impl<'a, T> Clone for BufferPool<'a, T> {
    /// Clones the handle (shared state), not the buffers. Cheap, like `Arc`.
    fn clone(&self) -> Self {
        BufferPool {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<'a, T> fmt::Debug for BufferPool<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `field` takes `&mut self` and returns `&mut DebugStruct`, so the
        // builder must be bound before the (feature-gated) `field` calls;
        // borrowed values are bound first to keep temporaries alive.
        let idle = self.idle_len();
        let mut debug = f.debug_struct("BufferPool");
        debug
            .field("idle", &idle)
            .field("max_idle", &self.inner.max_idle);
        #[cfg(feature = "stats")]
        {
            let stats = self.stats();
            debug.field("stats", &stats);
        }
        debug.finish_non_exhaustive()
    }
}

/// A leased buffer; the pool reclaims it when the guard is dropped.
///
/// This is a `#[must_use]` type: `pool.get()` without binding the result would
/// return the buffer immediately — almost certainly a mistake.
///
/// Use the buffer through [`Deref`]/[`DerefMut`] (`buf.iter_mut()`,
/// `buf.fill(0.0)`, `&mut *buf` to pass it on as `&mut Vec<_>`), and see
/// [`into_inner`](Pooled::into_inner) for detaching a buffer permanently.
///
/// Like the pool, the guard is `Send` whenever `T: Send`, so it may be moved
/// across threads mid-lease (the buffer returns to the pool from wherever it
/// is dropped).
///
/// # Example
///
/// ```
/// # use par_buffer_pool::BufferPool;
/// let pool = BufferPool::new(|| vec![1.0f64; 4]);
/// let mut buf = pool.get();
/// buf.push(2.0); // DerefMut to Vec<f64>, then to [f64]
/// assert_eq!(&*buf, &[1.0, 1.0, 1.0, 1.0, 2.0]);
/// // drop(buf) here would recycle it; scope end does the same
/// ```
#[must_use = "the buffer returns to the pool when the guard is dropped; bind it to use it"]
pub struct Pooled<'a, T> {
    buffer: Option<T>,
    inner: Arc<PoolInner<'a, T>>,
}

impl<'a, T> Pooled<'a, T> {
    /// Detaches the buffer: moves it out without returning it to the pool.
    ///
    /// Use this when the buffer becomes a *result* — it keeps its contents and
    /// will not be recycled (the pool will simply allocate a replacement on
    /// its next empty lease). For transient scratch, just drop the guard
    /// instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(Vec::new);
    /// let mut buf = pool.get();
    /// buf.push(42);
    /// let owned: Vec<i32> = buf.into_inner(); // detached
    /// assert_eq!(owned, vec![42]);
    /// assert_eq!(pool.idle_len(), 0); // nothing recycled
    /// ```
    pub fn into_inner(mut self) -> T {
        self.buffer
            .take()
            .expect("Pooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Deref for Pooled<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.buffer
            .as_ref()
            .expect("Pooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> DerefMut for Pooled<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.buffer
            .as_mut()
            .expect("Pooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Drop for Pooled<'a, T> {
    fn drop(&mut self) {
        // On `into_inner` there is nothing left to return; otherwise the
        // buffer is recycled (or dropped at the idle cap). Runs on early
        // returns and during panic unwinds just the same.
        if let Some(buffer) = self.buffer.take() {
            self.inner.recycle(buffer);
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for Pooled<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pooled")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_keeps_pool_alive_after_handles_are_dropped() {
        let pool = BufferPool::new(|| vec![0.0f64; 8]);
        let guard = pool.get();
        let orphan = pool.clone();
        drop(pool);
        drop(orphan);
        // The guard holds the last reference to the shared state; dropping it
        // must neither panic nor leak, even though no handle remains.
        drop(guard);
    }
}
