//! The shared, sharded pool: [`BufferPool`] with its [`SharedPooled`] guard.
//!
//! Idle buffers park in one of N cache-line-padded [`Mutex<Vec<T>>`] shards
//! (N = the machine's parallelism rounded up to a power of two, at least
//! 128), selected by the *parking* thread's dense id, plus one communal
//! reserve pile stocked by [`prefill`](BufferPool::prefill) and consulted
//! when a lease finds the shard empty. Any thread can still satisfy any
//! cold lease, and buffers return to the pool from whichever thread drops
//! them — but steady-state traffic (lease, use, return on the same worker)
//! only ever touches that worker's own lock, so many tiny tasks at high
//! worker counts do not convoy on a single mutex (the measured failure mode
//! of a one-pile pool; see the crate-level design notes). Initializers may
//! borrow non-`'static` data. For the per-thread, registry-based
//! alternative, see [`ThreadLocalPool`](crate::ThreadLocalPool)
//! ([`crate::local`]) and the crate-level
//! [choosing guide](crate#which-pool).

use std::cell::Cell;
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// Counters describing what a pool has done so far.
///
/// Returned by [`BufferPool::stats`](crate::BufferPool::stats) and
/// [`ThreadLocalPool::stats`](crate::ThreadLocalPool::stats), which are
/// behind the `stats` feature; the type itself is always available, so
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

/// Lower bound on a pool's shard count: 128 is roughly the largest thread
/// count sharing one NUMA domain on current top-end AMD servers, so the
/// floor puts every thread inside the most contended coherence domain on
/// its own shard — cross-thread shard sharing only begins at
/// NUMA-boundary scale, where cross-socket traffic dominates anyway.
/// Machines far below 128 workers pay the same fixed 16 KiB of padded
/// shard headers per pool and no more; the sizing never goes *below* this
/// so that behavior on a laptop and a 512-thread server differs only in
/// memory, never in contract.
const MIN_SHARDS: usize = 128;

/// Number of idle-buffer shards a new pool gets: the machine's parallelism
/// rounded up to a power of two (so the dense thread id maps to a shard by
/// masking), floored at [`MIN_SHARDS`]. One shard per worker keeps every
/// lease's lock effectively thread-private — the property that removes the
/// one-pile futex convoy — so sizing from
/// [`available_parallelism`](std::thread::available_parallelism) (which
/// respects CPU affinity and cgroup limits on Linux) tracks the machines
/// that actually run the code. Past the worker count extra shards buy
/// nothing (no threads left to separate), which is why this is not simply
/// "as many as possible". Costs `count * 128` bytes of padded shard headers
/// per pool: 16 KiB at the floor, 64 KiB on a 512-thread box — noise next
/// to the pooled buffers themselves.
fn shard_count() -> usize {
    std::thread::available_parallelism()
        .map_or(MIN_SHARDS, |n| n.get().next_power_of_two().max(MIN_SHARDS))
}

/// Monotonic source of dense thread ids. Ids are never reused, so a shard's
/// contents always belong to the same (possibly dead) thread population: a
/// thread that exits leaves its parked buffers in its shard until a
/// [`drain`](BufferPool::drain) or the pool's drop reclaims them.
static NEXT_TID: AtomicUsize = AtomicUsize::new(0);

// This thread's dense id, assigned on first use; `usize::MAX` marks
// "unassigned". One const-initialized TLS access per lease is the entire
// cost of sharding (measured below noise next to the lock it routes to).
thread_local! {
    static TID: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// Returns this thread's dense id, assigning one on the first call.
fn tid() -> usize {
    TID.with(|cell| {
        let assigned = cell.get();
        if assigned == usize::MAX {
            let fresh = NEXT_TID.fetch_add(1, Ordering::Relaxed);
            cell.set(fresh);
            fresh
        } else {
            assigned
        }
    })
}

/// One shard: an idle pile, padded to whole cache lines so that neighboring
/// shards' locks never share coherence traffic.
#[repr(align(128))]
struct Shard<T>(Mutex<Vec<T>>);

/// Locks an idle pile (a shard or the reserve), recovering from poisoning.
///
/// The lock only ever guards a `pop`/`push` of plain values, so a panic
/// elsewhere can leave it poisoned but never leaves damaged data behind;
/// recovering beats panicking on every later lease and beats leaking the
/// idle buffers.
fn lock_pile<T>(mutex: &Mutex<Vec<T>>) -> MutexGuard<'_, Vec<T>> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct PoolInner<'a, T> {
    shards: Box<[Shard<T>]>,
    /// `shards.len() - 1`, precomputed so the hot path masks instead of
    /// loading the length (and so the mask survives the `Box<[_]>` slice
    /// length living behind a pointer).
    shard_mask: usize,
    /// The communal cold-start pile: stocked by
    /// [`prefill`](BufferPool::prefill), drained by leases that miss their
    /// shard. Returns never push here (they park in the returning thread's
    /// shard), so the reserve cannot become a contention point; it exists so
    /// a buffer parked before the workers exist is found by whichever worker
    /// leases first.
    reserve: Mutex<Vec<T>>,
    init: InitFn<'a, T>,
    reset: Option<ResetFn<'a, T>>,
    max_idle: usize,
    /// Live count of parked buffers, maintained **only while a cap is
    /// installed** (`max_idle != usize::MAX`); `None` on the default
    /// uncapped pool, so the hot path pays nothing for the cap it does not
    /// have. When `Some`, every park and every take adjusts it, and
    /// [`with_max_idle`](BufferPool::with_max_idle) seeds it from an exact
    /// recount taken under exclusive access.
    parked: Option<AtomicUsize>,
    #[cfg(feature = "stats")]
    leases: AtomicUsize,
    #[cfg(feature = "stats")]
    allocations: AtomicUsize,
}

impl<'a, T> BufferPool<'a, T> {
    /// Creates a pool whose buffers are made by `init`.
    ///
    /// `init` runs lazily, only when a lease finds the pool empty — never
    /// eagerly (the one exception being [`prefill`](BufferPool::prefill),
    /// which opts into eagerness). It may borrow from the surroundings
    /// (hence `'a` instead of a `'static` bound). The pool starts empty with
    /// no idle cap; see [`with_max_idle`](BufferPool::with_max_idle) and
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
        let shards: Box<[Shard<T>]> = (0..shard_count())
            .map(|_| Shard(Mutex::new(Vec::new())))
            .collect();
        BufferPool {
            inner: Arc::new(PoolInner {
                shard_mask: shards.len() - 1,
                shards,
                reserve: Mutex::new(Vec::new()),
                init: Box::new(init),
                reset: None,
                max_idle: usize::MAX,
                parked: None,
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
    /// dropped instead of parked. The cap is global — a count over every
    /// shard and the reserve — and installing one adds one relaxed atomic
    /// adjustment per park/take to the lease path (an uncapped pool, the
    /// default, pays nothing). Any `max_idle` of at least the expected
    /// worker count keeps steady-state recycling intact; a cap exists so that
    /// a burst of concurrency cannot park its buffers in the pool forever.
    ///
    /// # Panics
    ///
    /// Panics if this pool's shared state has been cloned or otherwise
    /// shared: the cap is installed in place, on the assumption that a
    /// freshly built pool is uniquely owned. Build the pool fully (chaining
    /// this off [`new`](BufferPool::new)) before cloning or sharing handles.
    pub fn with_max_idle(mut self, max_idle: usize) -> Self {
        let inner = Arc::get_mut(&mut self.inner).expect("freshly built pool is uniquely owned");
        inner.max_idle = max_idle;
        // Exclusive access (the same guarantee the field writes above rely
        // on): recount what is already parked, so the counter starts exact
        // even if buffers were parked before the cap was installed. The
        // piles are locked one at a time, never nested.
        inner.parked = (max_idle != usize::MAX).then(|| AtomicUsize::new(inner.count_idle()));
        self
    }

    /// Registers `reset` to run each time a buffer is returned (builder
    /// style), so that every lease starts from a known state.
    ///
    /// `reset` runs when a [`SharedPooled`] guard is dropped and on manual
    /// [`put`](BufferPool::put) — i.e. on *every* path into the pool,
    /// including a return that the idle cap then discards (the hook runs
    /// before the cap is consulted, because user code must never run under
    /// a shard lock). The one drop path that skips the hook is
    /// [`drain`](BufferPool::drain), which hands buffers out of the pool
    /// rather than into it.
    ///
    /// Typical use: `|buf: &mut Vec<f64>| buf.fill(0.0)` for accumulation
    /// buffers. Skip the hook entirely when every consumer fully overwrites
    /// the buffer anyway — resets are not free.
    ///
    /// # Panics
    ///
    /// Panics if this pool's shared state has been cloned or otherwise
    /// shared: the hook is installed in place, on the assumption that a
    /// freshly built pool is uniquely owned. Build the pool fully (chaining
    /// [`with_reset`](BufferPool::with_reset) off [`new`](BufferPool::new))
    /// before cloning or sharing handles.
    pub fn with_reset(mut self, reset: impl Fn(&mut T) + Send + Sync + 'a) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("freshly built pool is uniquely owned")
            .reset = Some(Box::new(reset));
        self
    }

    /// Eagerly fills the pool with `n` fresh buffers (builder style), so a
    /// latency-sensitive phase does not pay the initializer inside its first
    /// parallel leases.
    ///
    /// The buffers are built by the same `init` closure [`new`](BufferPool::new)
    /// stores — this is the one place the initializer runs eagerly. Each
    /// buffer goes through the normal return-path checks: a reset hook
    /// registered with [`with_reset`](BufferPool::with_reset) runs, and a cap
    /// set with [`with_max_idle`](BufferPool::with_max_idle) limits how many
    /// are kept (chain it before this call if the fill should respect the
    /// cap; prefilling past the cap just drops the excess). The survivors
    /// park in the pool's communal reserve rather than the calling thread's
    /// shard — that is what makes a prefill servable by whichever workers
    /// lease first, on their very first lease.
    ///
    /// [`ThreadLocalPool`](crate::ThreadLocalPool) has no `prefill`: each
    /// thread warms its own slot with its first lease, and there is no
    /// cross-thread pile to fill from here.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(|| vec![0u8; 1024]).prefill(4);
    /// assert_eq!(pool.idle_len(), 4); // warm before the parallel phase
    ///
    /// let mut buf = pool.get(); // served from the prefill, no init run
    /// assert_eq!(buf.len(), 1024);
    /// # #[cfg(feature = "stats")]
    /// # assert_eq!(pool.stats().allocations, 4);
    /// ```
    pub fn prefill(self, n: usize) -> Self {
        for _ in 0..n {
            #[cfg(feature = "stats")]
            self.inner.allocations.fetch_add(1, Ordering::Relaxed);
            let buffer = (self.inner.init)();
            self.inner.stock_reserve(buffer);
        }
        self
    }

    /// Leases a buffer, returning a [`SharedPooled`] guard.
    ///
    /// Recycles an idle buffer when one is available — this thread's shard
    /// first, then the communal [`prefill`](BufferPool::prefill) reserve —
    /// otherwise runs the initializer (outside any lock). A lease can
    /// therefore run the initializer while some *other* thread's shard
    /// still holds idle buffers: sharding trades the momentary global view
    /// of a one-pile pool for per-worker locks, so the buffer count
    /// converges to the sum of per-shard peaks rather than the global peak
    /// (still bounded by [`with_max_idle`](BufferPool::with_max_idle), and
    /// still never losing a returned buffer). The buffer goes back to the
    /// pool when the guard is dropped — keep the guard for as long as the
    /// buffer is in use, e.g. bind it for the body of the task closure.
    ///
    /// The returned guard *borrows* the pool — no reference count is taken,
    /// which is most of this pool's per-lease overhead over the lock-free
    /// [`ThreadLocalPool`](crate::ThreadLocalPool). The pool handle must
    /// therefore outlive the guard (it nearly always does: the handle lives
    /// in the driver scope, the guard in a task closure). For the rare
    /// guard that must outlive every pool handle, [`get_owned`](BufferPool::get_owned)
    /// returns an owning guard that keeps the pool alive, at the cost of
    /// one `Arc` reference-count increment/decrement per lease.
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
    pub fn get(&self) -> SharedPooled<'_, 'a, T> {
        #[cfg(feature = "stats")]
        self.inner.leases.fetch_add(1, Ordering::Relaxed);
        // Each pile's guard is released at its own statement's semicolon:
        // the shard lock before the reserve is touched, both before the
        // initializer runs — allocation can take arbitrarily long and must
        // not stall any lease, and user code (the initializer) must never
        // run under a lock.
        let popped = self.inner.pop_shard();
        let buffer = match popped.or_else(|| self.inner.pop_reserve()) {
            Some(buffer) => buffer,
            None => {
                #[cfg(feature = "stats")]
                self.inner.allocations.fetch_add(1, Ordering::Relaxed);
                (self.inner.init)()
            }
        };
        SharedPooled {
            buffer: Some(buffer),
            pool: &self.inner,
        }
    }

    /// The closure-free twin of [`get`](BufferPool::get) for the rare guard
    /// that must outlive every pool handle: the returned
    /// [`SharedPooledOwned`] owns a handle to the pool's shared state, so
    /// the pool (and its buffers, initializer, and hooks) lives at least as
    /// long as the lease — even if every `BufferPool` value is dropped
    /// first. Dropping the guard recycles the buffer exactly as
    /// [`get`](BufferPool::get)'s does.
    ///
    /// The flexibility costs one `Arc` reference-count increment on lease
    /// and one decrement on return, on a cache line shared by all workers —
    /// measurable in the guard-per-task pattern (the reason it is *not*
    /// what plain [`get`](BufferPool::get) does). Prefer
    /// [`get`](BufferPool::get) whenever the pool handle's scope already
    /// encloses the guard's, which is the overwhelmingly common shape.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(|| vec![0u8; 4]);
    /// let mut buf = pool.get_owned(); // owns a handle: keeps the pool alive
    /// drop(pool);                     // every handle is gone…
    /// buf[0] = 7;                     // …the lease still works
    /// assert_eq!(buf[0], 7);
    /// // `buf` recycles into the (still alive) pool here
    /// ```
    pub fn get_owned(&self) -> SharedPooledOwned<'a, T> {
        // Same lease path as `get`; only the guard's ownership differs.
        // `take` (not a move — the lease implements `Drop`) leaves the
        // borrowed guard empty, so its own drop recycles nothing.
        let mut lease = self.get();
        SharedPooledOwned {
            buffer: lease.buffer.take(),
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
    /// runs here too. Like every return, the buffer parks in the *calling*
    /// thread's shard, so this thread's next lease is the one that finds it.
    pub fn put(&self, buffer: T) {
        self.inner.recycle(buffer);
    }

    /// Number of idle buffers currently parked in the pool, across every
    /// shard and the reserve. (Exact only absent in-flight leases, like
    /// every observation of a concurrent structure.)
    pub fn idle_len(&self) -> usize {
        self.inner.count_idle()
    }

    /// Removes every idle buffer from the pool, leaving it empty, and
    /// returns them as a [`Vec`], gathered pile by pile (each thread's
    /// shard, then the reserve) — parking order within a pile, pile order
    /// across the sweep.
    ///
    /// This is the explicit way to reclaim or hand off pooled storage while
    /// the pool lives on. (The passive alternatives are
    /// [`with_max_idle`](BufferPool::with_max_idle), which drops overflow on
    /// *return*, and dropping the pool itself.)
    ///
    /// **Call it between parallel phases, once every guard has been
    /// dropped.** Buffers checked out at that moment are not in the pile and
    /// therefore not in the returned [`Vec`]: they park again only when
    /// their guards drop, *after* the drain has already returned. Draining
    /// mid-phase does not fail in any way — it silently hands back a
    /// partial (possibly empty) pile and lets the pool refill as workers
    /// return, which is almost never what the caller intended. The moment a
    /// rayon `for_each`, a `thread::scope`, or any phase boundary ends is
    /// exactly the right time; anything earlier is not.
    ///
    /// The reset hook registered with
    /// [`with_reset`](BufferPool::with_reset) does *not* run — it belongs to
    /// the return path, and these buffers are leaving the pool (same
    /// semantics as [`SharedPooled::into_inner`]). The pool keeps working
    /// afterwards: its next lease finds it empty and runs the initializer.
    ///
    /// [`ThreadLocalPool`](crate::ThreadLocalPool) has no `drain`: its
    /// parked buffers sit in per-thread slots that no other thread can
    /// reach, so there is no pile to hand back — dropping the pool (or the
    /// parking thread exiting) is what reclaims them.
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
    ///
    /// Taken one step further, `drain` is the combine step of a parallel
    /// reduction — pooled accumulators hold the partial sums, `drain` hands
    /// them back, `.sum()` finishes; see the `drain_reduce` example.
    ///
    /// The caveat, made executable: a lease that is still out is simply
    /// absent — no panic, no error — and parks again afterwards.
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(|| vec![0u8; 8]);
    /// let held = pool.get(); // checked out: not in the pile
    /// drop(pool.get()); // a second buffer, parked
    /// let drained = pool.drain(); // best effort: only the parked one
    /// assert_eq!(drained.len(), 1);
    /// drop(held); // parks after the drain, refill begins
    /// assert_eq!(pool.idle_len(), 1);
    /// ```
    pub fn drain(&self) -> Vec<T> {
        // Gather pile by pile — each thread's shard, then the reserve —
        // locking one pile at a time and never nesting locks. Swapping in
        // an empty `Vec` needs no `T` bound: only the piles themselves are
        // defaulted, and the initializer is not involved. Under concurrent
        // leases the sweep is best-effort exactly as the single-pile
        // `mem::take` was: a buffer parked into an already-swept pile
        // stays behind (that is the documented mid-phase caveat above).
        let mut drained = Vec::new();
        let mut taken = 0usize;
        // `.iter()`, not `for .. in &self.inner.shards`: `&Box<[T]>` only
        // became `IntoIterator` in rustc 1.80, past the crate's MSRV.
        for shard in self.inner.shards.iter() {
            let pile = mem::take(&mut *lock_pile(&shard.0));
            taken += pile.len();
            drained.extend(pile);
        }
        let pile = mem::take(&mut *lock_pile(&self.inner.reserve));
        taken += pile.len();
        drained.extend(pile);
        if let Some(parked) = &self.inner.parked {
            // Keep the capped pool's count exact: subtract precisely what
            // left the piles, so racing parks (each already counted) stay
            // consistent.
            parked.fetch_sub(taken, Ordering::Relaxed);
        }
        drained
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
    /// This thread's shard, selected by its dense id. The mapping is sticky,
    /// so a thread's returns and its next leases hit the same pile — and the
    /// same lock — which is what keeps concurrent workers off each other's
    /// critical sections.
    fn shard(&self) -> &Shard<T> {
        &self.shards[tid() & self.shard_mask]
    }

    /// Pops from this thread's shard, adjusting the parked count when a cap
    /// is installed.
    fn pop_shard(&self) -> Option<T> {
        let popped = lock_pile(&self.shard().0).pop();
        if popped.is_some() {
            self.unpark(1);
        }
        popped
    }

    /// Pops from the communal reserve (the cold-start/prefill pile).
    fn pop_reserve(&self) -> Option<T> {
        let popped = lock_pile(&self.reserve).pop();
        if popped.is_some() {
            self.unpark(1);
        }
        popped
    }

    /// Records that `n` parked buffers left the piles — a no-op unless a cap
    /// is installed, in which case it is the other half of keeping the
    /// global count exact.
    fn unpark(&self, n: usize) {
        if let Some(parked) = &self.parked {
            parked.fetch_sub(n, Ordering::Relaxed);
        }
    }

    /// Whether one more buffer may park under the cap, claiming its slot in
    /// the parked count if so (and giving the slot back if not). Relaxed
    /// RMWs suffice: the only data the orderings would protect is the count
    /// itself, and the count is seeded under the exclusive access that
    /// `with_max_idle`'s builder contract guarantees.
    fn cap_allows(&self) -> bool {
        if let Some(parked) = &self.parked {
            if parked.fetch_add(1, Ordering::Relaxed) >= self.max_idle {
                parked.fetch_sub(1, Ordering::Relaxed);
                return false; // pool full — caller drops the buffer
            }
        }
        true
    }

    /// Parks `buffer` in the pool (resetting it first, if a hook is set),
    /// dropping it when the idle cap is reached: the return path behind
    /// guard drop and manual [`put`](BufferPool::put).
    fn recycle(&self, buffer: T) {
        let mut buffer = buffer;
        // The reset runs before the cap check, and outside the lock: user
        // code must never run under a shard lock, so a buffer discarded at
        // the cap still pays one hook call — the documented contract.
        if let Some(reset) = &self.reset {
            reset(&mut buffer);
        }
        if self.cap_allows() {
            lock_pile(&self.shard().0).push(buffer);
        }
        // else: pool full — `buffer` drops here, which is the point of the cap
    }

    /// Parks `buffer` in the communal reserve instead of a shard: the
    /// [`prefill`](BufferPool::prefill) path, so pre-filled buffers are
    /// findable by any worker's first lease. Same reset/cap contract as
    /// [`recycle`](PoolInner::recycle).
    fn stock_reserve(&self, buffer: T) {
        let mut buffer = buffer;
        if let Some(reset) = &self.reset {
            reset(&mut buffer);
        }
        if self.cap_allows() {
            lock_pile(&self.reserve).push(buffer);
        }
    }

    /// Sums parked buffers over every shard and the reserve, locking one
    /// pile at a time. Exact under exclusive access (how
    /// [`with_max_idle`](BufferPool::with_max_idle) seeds the counter);
    /// otherwise a best-effort observation, like any snapshot of concurrent
    /// state.
    fn count_idle(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| lock_pile(&shard.0).len())
            .sum::<usize>()
            + lock_pile(&self.reserve).len()
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

impl<'a, T: Default + 'a> Default for BufferPool<'a, T> {
    /// A pool whose initializer is [`Default::default`] — buffers are made
    /// by `T::default`, lazily, exactly as in [`BufferPool::new`].
    fn default() -> Self {
        BufferPool::new(T::default)
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
/// The guard *borrows* the pool for `'pool` — that is the whole point: a
/// lease takes no reference count, no atomics beyond the optional `stats`
/// counters, and pays only its shard's lock. The pool handle must outlive
/// the guard (the common shape: handle in the driver scope, guard in a task
/// closure); for the rare guard that must outlive every handle,
/// [`get_owned`](BufferPool::get_owned) returns [`SharedPooledOwned`].
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
pub struct SharedPooled<'pool, 'a, T> {
    pool: &'pool PoolInner<'a, T>,
    buffer: Option<T>,
}

impl<'pool, 'a, T> SharedPooled<'pool, 'a, T> {
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

impl<'pool, 'a, T> Deref for SharedPooled<'pool, 'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.buffer
            .as_ref()
            .expect("SharedPooled always holds its buffer until into_inner/drop")
    }
}

impl<'pool, 'a, T> DerefMut for SharedPooled<'pool, 'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.buffer
            .as_mut()
            .expect("SharedPooled always holds its buffer until into_inner/drop")
    }
}

impl<'pool, 'a, T> Drop for SharedPooled<'pool, 'a, T> {
    fn drop(&mut self) {
        // On `into_inner` there is nothing left to return; otherwise the
        // buffer is recycled (or dropped at the idle cap). Runs on early
        // returns and during panic unwinds just the same — and takes no
        // reference count: the borrow needs no release.
        if let Some(buffer) = self.buffer.take() {
            self.pool.recycle(buffer);
        }
    }
}

impl<'pool, 'a, T: fmt::Debug> fmt::Debug for SharedPooled<'pool, 'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPooled")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

/// A buffer leased through [`BufferPool::get_owned`]: the owning twin of
/// [`SharedPooled`]. Identical lease semantics — [`Deref`]/[`DerefMut`],
/// [`into_inner`](SharedPooledOwned::into_inner), recycle on drop — except
/// the guard owns a handle to the pool's shared state, so the pool outlives
/// the lease even if every `BufferPool` value has been dropped. The price
/// is one `Arc` reference-count increment/decrement per lease, on a cache
/// line shared by every worker; that is precisely what
/// [`get`](BufferPool::get)'s borrowing guard avoids.
///
/// Like [`SharedPooled`], the guard is `Send` whenever `T: Send`, so an
/// owning lease may be moved across threads mid-lease and still find its
/// way home afterwards.
///
/// # Example
///
/// ```
/// # use par_buffer_pool::BufferPool;
/// let pool = BufferPool::new(String::new);
/// let mut greeting = pool.get_owned();
/// drop(pool); // the pool lives on inside the guard
/// greeting.push_str("hello");
/// assert_eq!(&*greeting, "hello");
/// ```
#[must_use = "the buffer returns to the pool when the guard is dropped; bind it to use it"]
pub struct SharedPooledOwned<'a, T> {
    buffer: Option<T>,
    inner: Arc<PoolInner<'a, T>>,
}

impl<'a, T> SharedPooledOwned<'a, T> {
    /// Detaches the buffer: moves it out without returning it to the pool —
    /// same contract as [`SharedPooled::into_inner`]. The pool handle inside
    /// the guard is released here too: the pool dies once its last lease,
    /// handle, and parked buffer are gone.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::BufferPool;
    /// let pool = BufferPool::new(Vec::new);
    /// let mut buf = pool.get_owned();
    /// buf.push(42);
    /// let owned: Vec<i32> = buf.into_inner(); // detached
    /// assert_eq!(owned, vec![42]);
    /// ```
    pub fn into_inner(mut self) -> T {
        self.buffer
            .take()
            .expect("SharedPooledOwned always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Deref for SharedPooledOwned<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.buffer
            .as_ref()
            .expect("SharedPooledOwned always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> DerefMut for SharedPooledOwned<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.buffer
            .as_mut()
            .expect("SharedPooledOwned always holds its buffer until into_inner/drop")
    }
}

impl<'a, T> Drop for SharedPooledOwned<'a, T> {
    fn drop(&mut self) {
        // On `into_inner` there is nothing left to return; otherwise the
        // buffer is recycled (or dropped at the idle cap). The guard's own
        // `Arc` handle drops after this, releasing the pool if this was the
        // last reference.
        if let Some(buffer) = self.buffer.take() {
            self.inner.recycle(buffer);
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for SharedPooledOwned<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPooledOwned")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn shards_are_sized_to_the_machine() {
        // The sizing contract: power of two, at least MIN_SHARDS, and never
        // below the machine's parallelism (which available_parallelism
        // reports respecting affinity/cgroup limits).
        let n = std::thread::available_parallelism()
            .map_or(0, |n| n.get())
            .max(1);
        let expected = n.next_power_of_two().max(MIN_SHARDS);
        assert_eq!(shard_count(), expected);
        let pool = BufferPool::new(Vec::<u8>::new);
        assert_eq!(pool.inner.shards.len(), expected);
        assert_eq!(pool.inner.shard_mask, expected - 1);
    }

    #[test]
    fn owned_guard_keeps_pool_alive_after_handles_are_dropped() {
        let pool = BufferPool::new(|| vec![0.0f64; 8]);
        let guard = pool.get_owned();
        let orphan = pool.clone();
        drop(pool);
        drop(orphan);
        // The owned guard holds the last reference to the shared state;
        // dropping it must neither panic nor leak, even though no handle
        // remains — and it must recycle into the pool it kept alive.
        drop(guard);
    }

    #[test]
    fn prefill_in_the_reserve_serves_cold_leases_on_any_thread() {
        // Prefill parks in the communal reserve, not the prefilling
        // thread's shard, so a fresh worker's very first lease is served
        // without running the initializer — the whole point of prefill.
        let pool = BufferPool::new(|| vec![0u8; 8]).prefill(4);
        thread::scope(|s| {
            for _ in 0..4 {
                let pool = &pool;
                s.spawn(move || {
                    let buf = pool.get();
                    assert_eq!(buf.len(), 8);
                }); // guard drops here: parks in this worker's shard
            }
        });
        #[cfg(feature = "stats")]
        assert_eq!(pool.stats().allocations, 4, "no lease ran the initializer");
        assert_eq!(pool.idle_len(), 4, "every pre-filled buffer came back");
    }

    #[test]
    fn a_buffer_parks_in_the_thread_that_returns_it() {
        // The documented sharding trade: the worker's return parks in the
        // worker's shard, so the main thread's next lease — own shard and
        // reserve both empty — initializes fresh even though the pool is
        // not globally empty.
        let pool = BufferPool::new(|| vec![0u8; 4]);
        thread::scope(|s| {
            let pool = &pool;
            s.spawn(move || {
                drop(pool.get());
            });
        });
        assert_eq!(pool.idle_len(), 1, "parked in the worker's shard");
        drop(pool.get());
        #[cfg(feature = "stats")]
        assert_eq!(
            pool.stats().allocations,
            2,
            "main's lease initialized fresh"
        );
        assert_eq!(pool.idle_len(), 2, "one parked per shard now");
    }

    #[test]
    fn max_idle_is_global_across_shards() {
        // Returns from distinct threads land in distinct shards; the cap
        // still bounds the *total*, which is what the parked count keeps.
        let pool = BufferPool::new(|| vec![0u8; 2]).with_max_idle(3);
        thread::scope(|s| {
            for _ in 0..6 {
                let pool = &pool;
                s.spawn(move || {
                    drop(pool.get()); // six concurrent leases, six buffers
                });
            }
        }); // six returns race for three parking slots
        assert_eq!(pool.idle_len(), 3, "the cap bounds the pool total");
    }

    #[test]
    fn drain_gathers_every_shard_and_the_reserve() {
        let pool = BufferPool::new(|| vec![0u8; 2]).prefill(2);
        thread::scope(|s| {
            for _ in 0..2 {
                let pool = &pool;
                s.spawn(move || {
                    pool.put(vec![1u8; 2]); // parks in this worker's shard
                });
            }
        });
        assert_eq!(pool.idle_len(), 4, "two shards plus a stocked reserve");
        assert_eq!(pool.drain().len(), 4, "nothing strands in any pile");
        assert_eq!(pool.idle_len(), 0);
    }
}
