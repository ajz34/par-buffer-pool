//! The shared, mutex-guarded pool: [`BufferPool`] with its [`SharedPooled`]
//! guard.
//!
//! One [`Mutex`]-protected `Vec<T>` holds parked buffers, so any thread can
//! satisfy any lease and buffers return to the pool from whichever thread
//! drops them. Initializers may borrow non-`'static` data. For the
//! lock-free, per-thread alternative, see [`ThreadLocalPool`]
//! ([`crate::local`]) and the crate-level [choosing guide](crate#which-pool).

use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
#[cfg(feature = "stats")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Counters describing what a pool has done so far.
///
/// Returned by [`BufferPool::stats`](crate::BufferPool::stats) and
/// [`ThreadLocalPool::stats`](crate::ThreadLocalPool::stats), which
/// are behind the `stats` feature; the type itself is always available, so
/// generic code can name it without feature gates. The difference
/// `leases - allocations` is the number of leases served from recycled
/// buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// Total number of leases taken via [`BufferPool::get`] (or
    /// [`ThreadLocalPool::get`](crate::ThreadLocalPool::get)).
    pub leases: usize,
    /// Number of times the initializer had to run, i.e. buffers created from
    /// scratch because the pool was momentarily empty.
    ///
    /// This is an upper bound on the number of concurrently outstanding
    /// leases, so `allocations * buffer_size` bounds the scratch memory the
    /// pool can be responsible for.
    pub allocations: usize,
}

impl PoolStats {
    /// Leases served from an idle (recycled) buffer: `leases - allocations`.
    pub fn reuses(&self) -> usize {
        self.leases - self.allocations
    }
}

/// A thread-safe pool of reusable buffers of type `T`, shared by all threads.
///
/// Call [`get`](BufferPool::get) to lease a buffer as a [`SharedPooled`]
/// guard; the buffer returns itself to the pool when the guard is dropped.
/// See the [crate documentation](crate) for the full story, and
/// [`ThreadLocalPool`](crate::ThreadLocalPool) for the per-thread, lock-free
/// variant.
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
    /// `reset` runs when a [`SharedPooled`] guard is dropped and on manual
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

    /// Leases a buffer, returning a [`SharedPooled`] guard.
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
    pub fn get(&self) -> SharedPooled<'a, T> {
        #[cfg(feature = "stats")]
        self.inner.leases.fetch_add(1, Ordering::Relaxed);
        // Pop under the lock, but run the initializer outside it: allocation
        // can take arbitrarily long and must not stall other leases.
        let buffer = self.inner.lock_idle().pop().unwrap_or_else(|| {
            #[cfg(feature = "stats")]
            self.inner.allocations.fetch_add(1, Ordering::Relaxed);
            (self.inner.init)()
        });
        SharedPooled {
            buffer: Some(buffer),
            inner: Arc::clone(&self.inner),
        }
    }

    /// Runs `f` with a buffer checked out: the closure-based twin of
    /// [`get`](BufferPool::get), for call sites where a named guard is
    /// awkward. The buffer is out of the pool while `f` runs and returned
    /// afterwards — on early return, on panic unwind, and on the happy path
    /// alike.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(|| vec![0.0f64; 64]);
    /// let total: f64 = pool.with(|buf| {
    ///     buf.fill(1.0);
    ///     buf.iter().sum()
    /// }); // returned to the pool here
    /// assert_eq!(total, 64.0);
    /// assert_eq!(pool.idle_len(), 1);
    /// ```
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut lease = self.get();
        f(&mut lease)
    }

    /// Returns a raw buffer to the pool, dropping it if the pool is full.
    ///
    /// This is the manual escape hatch, for values that were detached with
    /// [`SharedPooled::into_inner`] earlier or produced independently; the
    /// normal path is simply dropping a [`SharedPooled`] guard. If a reset
    /// hook was registered with [`with_reset`](BufferPool::with_reset), it
    /// runs here too.
    pub fn put(&self, buffer: T) {
        self.inner.recycle(buffer);
    }

    /// Number of idle buffers currently parked in the pool.
    pub fn idle_len(&self) -> usize {
        self.inner.lock_idle().len()
    }

    /// Removes every idle buffer from the pool, leaving it empty, and
    /// returns them as a [`Vec`] in the order they were parked.
    ///
    /// This is the explicit way to reclaim or hand off pooled storage while
    /// the pool lives on — typically called from one thread between parallel
    /// phases, once every guard has been dropped. (The passive alternatives
    /// are [`with_max_idle`](BufferPool::with_max_idle), which drops
    /// overflow on *return*, and dropping the pool itself.)
    ///
    /// Outstanding leases are not affected: buffers checked out at this
    /// moment are not in the pile, and they simply park again when their
    /// guards drop. The reset hook registered with
    /// [`with_reset`](BufferPool::with_reset) does *not* run — it belongs to
    /// the return path, and these buffers are leaving the pool (same
    /// semantics as [`SharedPooled::into_inner`]). The pool keeps working
    /// afterwards: its next lease finds it empty and runs the initializer.
    ///
    /// On a [`ThreadLocalPool`](crate::ThreadLocalPool) the same method is
    /// thread-scoped: there is no shared pile, so it can only take the
    /// calling thread's parked buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(|| vec![0u8; 1024]);
    /// let holders: Vec<_> = (0..4).map(|_| pool.get()).collect();
    /// drop(holders); // four buffers parked
    /// let drained: Vec<Vec<u8>> = pool.drain();
    /// assert_eq!(drained.len(), 4);
    /// assert_eq!(pool.idle_len(), 0); // empty; the next lease allocates fresh
    /// ```
    ///
    /// The typical phase shape: workers lease, fill, and return scratch;
    /// afterwards one call on the main thread takes the whole pile back as
    /// owned buffers.
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// # use rayon::prelude::*;
    /// let pool = BufferPool::new(|| vec![0u8; 16]);
    ///
    /// (0..8u8).into_par_iter().for_each(|i| {
    ///     let mut buf = pool.get();
    ///     buf.fill(i);
    /// }); // every guard returned: the pile holds all the pool's buffers
    ///
    /// let pile: Vec<Vec<u8>> = pool.drain();
    /// assert!(!pile.is_empty()); // rayon decides how many were allocated
    /// assert!(pile.iter().all(|buf| buf.len() == 16));
    /// assert!(pile.iter().all(|buf| buf.iter().all(|&b| b == buf[0])));
    /// assert_eq!(pool.idle_len(), 0);
    /// ```
    pub fn drain(&self) -> Vec<T> {
        // Swapping in an empty `Vec` needs no `T` bound: only the pile
        // itself is defaulted. The initializer is not involved, so this can
        // stay a single locked step, like a `pop`.
        mem::take(&mut *self.inner.lock_idle())
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

/// A buffer leased from a [`BufferPool`]; the pool reclaims it when the
/// guard is dropped.
///
/// This is a `#[must_use]` type: `pool.get()` without binding the result
/// would return the buffer immediately — almost certainly a mistake.
///
/// Use the buffer through [`Deref`]/[`DerefMut`] (`buf.iter_mut()`,
/// `buf.fill(0.0)`, `&mut *buf` to pass it on as `&mut Vec<_>`), and see
/// [`into_inner`](SharedPooled::into_inner) for detaching a buffer
/// permanently.
///
/// Like the pool, the guard is `Send` whenever `T: Send`, so it may be moved
/// across threads mid-lease (the buffer returns to the pool from wherever it
/// is dropped). The guard keeps the pool alive even if all `BufferPool`
/// handles are dropped.
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
pub struct SharedPooled<'a, T> {
    buffer: Option<T>,
    inner: Arc<PoolInner<'a, T>>,
}

impl<'a, T> SharedPooled<'a, T> {
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
            .expect("SharedPooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Deref for SharedPooled<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.buffer
            .as_ref()
            .expect("SharedPooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> DerefMut for SharedPooled<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.buffer
            .as_mut()
            .expect("SharedPooled always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Drop for SharedPooled<'a, T> {
    fn drop(&mut self) {
        // On `into_inner` there is nothing left to return; otherwise the
        // buffer is recycled (or dropped at the idle cap). Runs on early
        // returns and during panic unwinds just the same.
        if let Some(buffer) = self.buffer.take() {
            self.inner.recycle(buffer);
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for SharedPooled<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPooled")
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
