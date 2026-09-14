//! The per-thread, lock-free pool: [`ThreadLocalPool`] with its
//! [`LocalPooled`] guard.
//!
//! Buffers live in thread-local storage — one lazily created slot per worker
//! thread per pool — so a lease never touches a lock or a shared cache line.
//! This is the structured answer to hand-rolled `thread_local!` +
//! `RefCell` scratch: the same ~ns access cost, plus the guard-based RAII
//! return, reset hooks, detach, and stats that raw TLS cannot give you.
//! Compared to [`BufferPool`](crate::BufferPool) (see the crate-level
//! [choosing guide](crate#which-pool)):
//!
//! - initializers must be `'static` (TLS values cannot borrow stack locals),
//! - each thread parks at most one buffer per pool (so peak memory is
//!   `nthreads * npools` even if only two threads are busy),
//! - a guard dropped on a *different* thread migrates the buffer to that
//!   thread's slot (there is no global pile to return into),
//! - and when a pool is dropped, its parked buffers are reclaimed lazily but
//!   reliably: each thread discards them on its next pool interaction, and
//!   thread exit is the hard backstop. Memory lingers only on threads that
//!   never touch any [`ThreadLocalPool`] again while staying alive.

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "stats")]
use crate::buffer::PoolStats;

/// A parked buffer, type-erased so that one concrete thread-local registry
/// can serve pools of every element type. Slots are written and read back
/// only through the pool that owns them, and pool ids are never reused, so
/// the erased type is always the owning pool's `T`.
type Erased = Box<dyn Any + Send>;

/// Monotonic source of pool ids. Ids are never reused, which lets a parked
/// slot trust that "same id" always means "same pool" — and lets one shared
/// registry hold every pool's slot without type information.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Bumped (with `Release`) every time any `ThreadLocalPool` is dropped. Each
/// thread's registry caches the value it last saw; a mismatch tells the
/// thread "some pool you may have parked buffers for is dead — check your
/// slots". One `Acquire` load per registry access is the entire reclamation
/// cost on the hot path; the line is read-only until a pool actually dies,
/// so it stays shared-read in every core's cache.
static DROP_EPOCH: AtomicUsize = AtomicUsize::new(0);

/// This thread's parked buffer for one pool, plus the liveness flag shared
/// with that pool. `live` flips to `false` when the pool is dropped; the
/// slot holds its own `Arc` clone, so the flag stays valid for the slot's
/// whole life without keeping the pool (its initializer, its hooks) alive.
struct Slot {
    live: Arc<AtomicBool>,
    buf: Option<Erased>,
}

/// A thread's registry: one slot per pool this thread has ever touched,
/// indexed by dense pool id, plus the drop-epoch this registry last swept.
struct Registry {
    seen_epoch: usize,
    slots: Vec<Slot>,
}

// This thread's registry, for every `ThreadLocalPool` of every element type
// (one concrete `thread_local!`; buffers are type-erased per slot).
//
// Deliberately NOT a `const {}` initializer: the non-const form pays one
// lazy-init branch per access (~ns, and gone after first use), and it
// registers a thread-exit destructor — which is the reclamation backstop.
// When a thread dies, its registry drops, releasing every parked buffer it
// still holds.
thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry {
        seen_epoch: 0,
        slots: Vec::new(),
    });
}

/// Runs `f` with this thread's registry, first performing the lazy
/// reclamation sweep if any pool was dropped since the last access on this
/// thread.
fn with_registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    REGISTRY.with(|cell| {
        let mut reg = cell.borrow_mut();
        let epoch = DROP_EPOCH.load(Ordering::Acquire);
        if epoch != reg.seen_epoch {
            reg.seen_epoch = epoch;
            for slot in &mut reg.slots {
                if slot.buf.is_some() && !slot.live.load(Ordering::Relaxed) {
                    // The pool is gone: drop its buffer (running `T`'s own
                    // destructor, no reset hook — the pool that owned the
                    // hook no longer exists). Mirrors `BufferPool`'s
                    // `with_max_idle` overflow, which also drops unreset.
                    slot.buf = None;
                }
            }
        }
        f(&mut reg)
    })
}

/// Returns this thread's slot for pool `id`, creating or repairing it as
/// needed.
fn slot_for<'s>(reg: &'s mut Registry, id: usize, live: &Arc<AtomicBool>) -> &'s mut Slot {
    if reg.slots.len() <= id {
        // First touch of this pool on this thread: grow to the dense id.
        reg.slots.resize_with(id + 1, || Slot { live: Arc::clone(live), buf: None });
        return &mut reg.slots[id];
    }
    let slot = &mut reg.slots[id];
    if !Arc::ptr_eq(&slot.live, live) {
        // The slot exists but was gap-filled by another pool's resize (ids
        // are never reused, so a mismatch can never hide a live buffer).
        debug_assert!(slot.buf.is_none());
        slot.live = Arc::clone(live);
    }
    slot
}

/// A per-thread pool of reusable buffers of type `T`: the lock-free sibling
/// of [`BufferPool`](crate::BufferPool).
///
/// Each worker thread lazily gets its own buffer slot; a lease on a thread
/// that already returned its buffer recycles it with no synchronization at
/// all — no lock, no shared cache line. Under the same workloads that drive
/// `BufferPool`'s shared mutex into futex-convoy territory (many tiny tasks
/// at high worker counts), `ThreadLocalPool` keeps scaling linearly. See the
/// crate-level [choosing guide](crate#which-pool) for when to prefer which.
///
/// The handle is [`Clone`] + [`Send`] + [`Sync`], so it is shared with rayon
/// or scoped threads by reference, exactly like `BufferPool`. The
/// differences from `BufferPool`: `'static` initializers, one buffer per
/// thread, migrating buffers, lazy-but-reliable reclamation after drop, and
/// no idle cap (there is no global pile to cap — each thread parks at most
/// one).
///
/// # Example
///
/// ```
/// use par_buffer_pool::ThreadLocalPool;
/// use rayon::prelude::*;
///
/// // built once before the parallel loop, exactly like a BufferPool
/// let pool = ThreadLocalPool::new(|| vec![0.0f64; 1 << 20]);
///
/// let sums: Vec<f64> = (0..64)
///     .into_par_iter()
///     .map(|task| {
///         let mut buf = pool.get(); // per-thread lease: no lock taken
///         buf.fill(task as f64);
///         buf.iter().sum() // returned to *this thread's* slot at scope end
///     })
///     .collect();
///
/// # #[cfg(feature = "stats")]
/// # {
/// let stats = pool.stats();
/// assert_eq!(stats.leases, 64);
/// # }
/// ```
pub struct ThreadLocalPool<T> {
    /// Dense identity of this pool within every thread's registry.
    id: usize,
    inner: Arc<LocalInner<T>>,
}

struct LocalInner<T> {
    init: Box<dyn Fn() -> T + Send + Sync>,
    reset: Option<Box<dyn Fn(&mut T) + Send + Sync>>,
    /// Shared liveness flag; every parked slot holds a clone. False once this
    /// pool is dropped.
    live: Arc<AtomicBool>,
    #[cfg(feature = "stats")]
    leases: AtomicUsize,
    #[cfg(feature = "stats")]
    allocations: AtomicUsize,
}

impl<T> Drop for LocalInner<T> {
    fn drop(&mut self) {
        // Announce death: parked slots see the flag on their next sweep and
        // release their buffers. The epoch bump is what wakes the sweeps.
        self.live.store(false, Ordering::Release);
        DROP_EPOCH.fetch_add(1, Ordering::Release);
    }
}

impl<T: Send + 'static> ThreadLocalPool<T> {
    /// Creates a pool whose buffers are made by `init`.
    ///
    /// Unlike [`BufferPool::new`](crate::BufferPool::new), `init` must be
    /// `'static`: thread-local slots cannot borrow from the caller's frame.
    /// The pool starts with no per-thread slots; each thread's buffer is
    /// created on its first lease.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let n = 256;
    /// let pool = ThreadLocalPool::new(move || vec![0.0f64; n]); // `move`: 'static
    /// assert_eq!(pool.get().len(), 256);
    /// ```
    pub fn new(init: impl Fn() -> T + Send + Sync + 'static) -> Self {
        ThreadLocalPool {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            inner: Arc::new(LocalInner {
                init: Box::new(init),
                reset: None,
                live: Arc::new(AtomicBool::new(true)),
                #[cfg(feature = "stats")]
                leases: AtomicUsize::new(0),
                #[cfg(feature = "stats")]
                allocations: AtomicUsize::new(0),
            }),
        }
    }

    /// Registers `reset` to run each time a buffer is returned (builder
    /// style), so that every lease starts from a known state — same contract
    /// as [`BufferPool::with_reset`](crate::BufferPool::with_reset). The hook
    /// runs when a [`LocalPooled`] guard is dropped and on manual
    /// [`put`](ThreadLocalPool::put). Buffers reclaimed after the pool is
    /// dropped are dropped without being reset (the hook is gone by then).
    pub fn with_reset(mut self, reset: impl Fn(&mut T) + Send + Sync + 'static) -> Self {
        let inner = Arc::get_mut(&mut self.inner)
            .expect("freshly built pool is uniquely owned");
        inner.reset = Some(Box::new(reset));
        self
    }

    /// Leases this thread's buffer, returning a [`LocalPooled`] guard.
    ///
    /// Recycles the thread's parked buffer when it has one (the common path:
    /// no lock, no atomics), otherwise runs the initializer. The buffer
    /// parks on *whichever thread's* slot the guard is dropped on, so keep
    /// the guard bound in the task closure that uses it.
    ///
    /// The guard borrows the pool (`LocalPooled<'_, T>`): unlike
    /// [`SharedPooled`](crate::SharedPooled) it does not keep the pool alive,
    /// which is what lets a lease avoid touching any reference count. Keep
    /// the handle alive for the guard's scope (it nearly always is).
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let pool = ThreadLocalPool::new(String::new);
    /// let mut greeting = pool.get();
    /// greeting.push_str("hello");
    /// assert_eq!(&*greeting, "hello");
    /// ```
    pub fn get(&self) -> LocalPooled<'_, T> {
        #[cfg(feature = "stats")]
        self.inner.leases.fetch_add(1, Ordering::Relaxed);
        // Take out of the slot under the (thread-private) registry borrow,
        // but run the initializer outside it, like BufferPool runs its
        // initializer outside the mutex.
        let leased = with_registry(|reg| slot_for(reg, self.id, &self.inner.live).buf.take());
        let buffer = match leased {
            Some(erased) => match erased.downcast::<T>() {
                Ok(boxed) => boxed,
                Err(erased) => {
                    // Unreachable by construction: slots are written only by
                    // their owning pool, and ids are never reused. Park the
                    // foreign buffer back before failing loudly, so a bug
                    // here cannot silently eat another pool's scratch.
                    with_registry(|reg| slot_for(reg, self.id, &self.inner.live).buf = Some(erased));
                    panic!("ThreadLocalPool slot holds a buffer of the wrong type");
                }
            },
            None => {
                #[cfg(feature = "stats")]
                self.inner.allocations.fetch_add(1, Ordering::Relaxed);
                Box::new((self.inner.init)())
            }
        };
        LocalPooled {
            pool: self,
            buffer: Some(buffer),
        }
    }

    /// Runs `f` with this thread's buffer checked out: the closure-based
    /// twin of [`get`](ThreadLocalPool::get), for call sites where a named
    /// guard is awkward. The buffer is checked out (slot empty) while `f`
    /// runs and parked again afterwards — on early return, on panic unwind,
    /// and on the happy path alike. A nested [`get`](ThreadLocalPool::get) or
    /// [`with`](ThreadLocalPool::with) while a buffer is checked out simply
    /// allocates a fresh one, exactly like leasing twice from a
    /// [`BufferPool`](crate::BufferPool); nothing panics.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let pool = ThreadLocalPool::new(|| vec![0.0f64; 64]);
    /// let total: f64 = pool.with(|buf| {
    ///     buf.fill(1.0);
    ///     buf.iter().sum()
    /// }); // returned to this thread's slot here
    /// assert_eq!(total, 64.0);
    /// ```
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut lease = self.get();
        f(&mut lease)
    }

    /// Returns a raw buffer to *this thread's* slot for this pool.
    ///
    /// The manual escape hatch for values detached with
    /// [`into_inner`](LocalPooled::into_inner) earlier or produced
    /// independently; the normal path is dropping a [`LocalPooled`] guard.
    /// A reset hook registered with
    /// [`with_reset`](ThreadLocalPool::with_reset) runs here too. Note the
    /// asymmetry with [`BufferPool::put`](crate::BufferPool::put): there is
    /// no global pile, so the buffer parks on the *calling* thread.
    pub fn put(&self, buffer: T) {
        self.park(buffer);
    }

    /// Parks an already-boxed buffer. The guard's drop path lands here via
    /// `Box<T> → Box<dyn Any + Send>` coercion — a pointer cast, no
    /// allocation — which keeps the lease round-trip malloc-free.
    fn park_erased(&self, mut buffer: Erased) {
        if let Some(reset) = &self.inner.reset {
            // Safety of the downcast: slots are written only by their owning
            // pool, and ids are never reused, so this pool's slot always
            // holds this pool's `T`. Same invariant as in `get`.
            let typed = buffer.downcast_mut::<T>().expect(
                "ThreadLocalPool slot holds a buffer of the wrong type",
            );
            reset(typed);
        }
        with_registry(|reg| slot_for(reg, self.id, &self.inner.live).buf = Some(buffer));
    }

    /// Whether *this thread* currently has a buffer parked for this pool:
    /// `0` or `1`. Thread-local pools have no global idle pile to count
    /// (each thread parks at most one buffer), so unlike
    /// [`BufferPool::idle_len`](crate::BufferPool::idle_len) this is a
    /// thread-scoped fact — useful in tests and assertions.
    pub fn idle_len(&self) -> usize {
        with_registry(|reg| {
            reg.slots
                .get(self.id)
                .map_or(0, |slot| slot.buf.is_some() as usize)
        })
    }

    /// Takes this thread's parked buffer out of the pool, returning it as a
    /// one-element [`Vec`] — or an empty one when this thread has nothing
    /// parked (including never having leased at all).
    ///
    /// Thread-scoped exactly like [`idle_len`](ThreadLocalPool::idle_len): a
    /// thread-local pool has no global pile, so buffers parked on *other*
    /// threads are not reachable from here. Collecting worker scratch on the
    /// main thread therefore needs the guards themselves (send them back
    /// mid-lease and [`into_inner`](LocalPooled::into_inner) them);
    /// worker-parked buffers live on in their slots until that thread leases
    /// again, the pool is dropped, or the thread exits.
    ///
    /// The reset hook registered with
    /// [`with_reset`](ThreadLocalPool::with_reset) does *not* run — it
    /// belongs to the return path, and this buffer is leaving the pool (same
    /// semantics as [`into_inner`](LocalPooled::into_inner)). The pool keeps
    /// working: this thread's next lease finds an empty slot and runs the
    /// initializer.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let pool = ThreadLocalPool::new(|| vec![0u8; 1024]);
    /// drop(pool.get()); // parks on this thread
    /// let drained: Vec<Vec<u8>> = pool.drain();
    /// assert_eq!(drained.len(), 1);
    /// assert_eq!(pool.idle_len(), 0);
    /// assert_eq!(pool.drain().len(), 0); // nothing further parked here
    /// ```
    pub fn drain(&self) -> Vec<T> {
        // `get_mut`, not `slot_for`: like `idle_len`, draining must not
        // create a slot. The sweep inside `with_registry` has already
        // dropped the buffer if the pool is dead.
        let leased =
            with_registry(|reg| reg.slots.get_mut(self.id).and_then(|slot| slot.buf.take()));
        match leased {
            Some(erased) => match erased.downcast::<T>() {
                Ok(boxed) => vec![*boxed],
                Err(erased) => {
                    // Unreachable by construction (same invariant as `get`);
                    // park the foreign buffer back before failing loudly.
                    with_registry(|reg| {
                        slot_for(reg, self.id, &self.inner.live).buf = Some(erased)
                    });
                    panic!("ThreadLocalPool slot holds a buffer of the wrong type");
                }
            },
            None => Vec::new(),
        }
    }

    /// Lease and allocation counters; see [`PoolStats`]. Same contract and
    /// same `stats` feature gate as
    /// [`BufferPool::stats`](crate::BufferPool::stats): the counters are
    /// shared atomics, so enabling the feature re-adds one relaxed add on a
    /// shared cache line per lease — off by default precisely so the
    /// lock-free path stays lock-free.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let pool = ThreadLocalPool::new(Vec::<u8>::new);
    /// for k in 0..10 {
    ///     drop(pool.get()); // returned, so the next lease recycles it
    ///     let _ = k;
    /// }
    /// let stats = pool.stats();
    /// # assert_eq!(stats.leases, 10);
    /// assert_eq!(stats.allocations, 1); // one buffer served all ten leases
    /// ```
    #[cfg(feature = "stats")]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            leases: self.inner.leases.load(Ordering::Relaxed),
            allocations: self.inner.allocations.load(Ordering::Relaxed),
        }
    }

    /// Parks `buffer` in this thread's slot (resetting it first, if a hook
    /// is set). The slot is created on demand — a guard that migrated from
    /// another thread parks here even if this thread never leased before.
    /// The slot holds a single buffer: parking over a parked one (possible
    /// only via migration or nested leases) drops the previous buffer.
    fn park(&self, buffer: T) {
        self.park_erased(Box::new(buffer));
    }
}

impl<T> Clone for ThreadLocalPool<T> {
    /// Clones the handle (shared state), not the buffers. Cheap, like `Arc`.
    fn clone(&self) -> Self {
        ThreadLocalPool {
            id: self.id,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> fmt::Debug for ThreadLocalPool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // No `idle` field: counting parked buffers requires the `T: Send +
        // 'static` bound and would touch the current thread's registry.
        let mut debug = f.debug_struct("ThreadLocalPool");
        debug.field("id", &self.id);
        #[cfg(feature = "stats")]
        {
            let stats = PoolStats {
                leases: self.inner.leases.load(Ordering::Relaxed),
                allocations: self.inner.allocations.load(Ordering::Relaxed),
            };
            debug.field("stats", &stats);
        }
        debug.finish_non_exhaustive()
    }
}

/// A buffer leased from a [`ThreadLocalPool`]; it returns to the *dropping*
/// thread's slot when the guard is dropped.
///
/// This is a `#[must_use]` type: `pool.get()` without binding the result
/// would return the buffer immediately — almost certainly a mistake.
///
/// Use the buffer through [`Deref`]/[`DerefMut`] (`buf.iter_mut()`,
/// `buf.fill(0.0)`, `&mut *buf` to pass it on as `&mut Vec<_>`), and see
/// [`into_inner`](LocalPooled::into_inner) for detaching a buffer
/// permanently.
///
/// The guard is `Send` whenever `T: Send` (the pool handle it borrows is
/// `Sync`), so it may be moved across threads mid-lease — the buffer then
/// parks on the thread that drops it. Unlike
/// [`SharedPooled`](crate::SharedPooled), the guard borrows the pool instead
/// of keeping it alive; the borrow is what keeps the hot path free of atomics.
///
/// # Example
///
/// ```
/// # use par_buffer_pool::ThreadLocalPool;
/// let pool = ThreadLocalPool::new(|| vec![1.0f64; 4]);
/// let mut buf = pool.get();
/// buf.push(2.0);
/// assert_eq!(&*buf, &[1.0, 1.0, 1.0, 1.0, 2.0]);
/// // drop(buf) here would park it on this thread; scope end does the same
/// ```
// The struct carries the pool's own bounds (`T: Send + 'static`) because a
// `Drop` impl may not add bounds: parking the buffer on guard drop erases it
// into a `Box<dyn Any + Send>`, which needs both.
#[must_use = "the buffer returns to the pool when the guard is dropped; bind it to use it"]
pub struct LocalPooled<'a, T: Send + 'static> {
    pool: &'a ThreadLocalPool<T>,
    buffer: Option<Box<T>>,
}

impl<T: Send + 'static> LocalPooled<'_, T> {
    /// Detaches the buffer: moves it out without returning it to the pool.
    ///
    /// Use this when the buffer becomes a *result* — it keeps its contents
    /// and will not be recycled (the slot stays empty and the thread's next
    /// lease runs the initializer again). For transient scratch, just drop
    /// the guard instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use par_buffer_pool::ThreadLocalPool;
    /// let pool = ThreadLocalPool::new(Vec::<i32>::new);
    /// let mut buf = pool.get();
    /// buf.push(42);
    /// let owned: Vec<i32> = buf.into_inner(); // detached
    /// assert_eq!(owned, vec![42]);
    /// assert_eq!(pool.idle_len(), 0); // nothing parked
    /// ```
    pub fn into_inner(mut self) -> T {
        *self
            .buffer
            .take()
            .expect("LocalPooled always holds its buffer until into_inner/drop")
    }
}

impl<T: Send + 'static> Drop for LocalPooled<'_, T> {
    fn drop(&mut self) {
        // On `into_inner` there is nothing left to return; otherwise park on
        // this thread — even if the lease started elsewhere. The `Box<T>`
        // coerces to the erased slot type without reallocating. Runs on
        // early returns and during panic unwinds just the same.
        if let Some(buffer) = self.buffer.take() {
            self.pool.park_erased(buffer);
        }
    }
}

impl<T: Send + 'static> Deref for LocalPooled<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.buffer
            .as_ref()
            .expect("LocalPooled always holds its buffer until into_inner/drop")
    }
}

impl<T: Send + 'static> DerefMut for LocalPooled<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.buffer
            .as_mut()
            .expect("LocalPooled always holds its buffer until into_inner/drop")
    }
}

impl<T: Send + 'static + fmt::Debug> fmt::Debug for LocalPooled<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalPooled")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parked_buffer_is_thread_scoped_and_recycled() {
        let pool = ThreadLocalPool::new(|| vec![0u8; 8]);
        {
            let mut buf = pool.get();
            buf[0] = 42;
        }
        assert_eq!(pool.idle_len(), 1);
        let recycled = pool.get();
        assert_eq!(recycled[0], 42);
    }

    #[test]
    fn reclaiming_a_dropped_pool_frees_its_parked_buffer() {
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&drops);
        let dead = ThreadLocalPool::new(move || Tracked(Arc::clone(&counter)));
        drop(dead.get()); // parked on this thread
        drop(dead); // pool gone; the parked Tracked now outlives its pool

        assert_eq!(drops.load(Ordering::SeqCst), 0, "parked until a sweep");
        // Any access to the registry performs the sweep.
        let other = ThreadLocalPool::new(|| ());
        let _ = other.idle_len();
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "sweep released the dead pool's parked buffer"
        );
    }
}
