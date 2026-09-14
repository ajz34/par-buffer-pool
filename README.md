# par-buffer-pool

A tiny, dependency-free, `#![forbid(unsafe_code)]` crate with **two buffer
pools** with **RAII guards**, for reusing scratch buffers across parallel
workers — rayon, scoped threads, or a single thread:

- **[`BufferPool`]** — one shared, mutex-guarded pile; any thread can satisfy
  any lease. Initializers may borrow non-`'static` data.
- **[`ThreadLocalPool`]** — one slot per worker thread; leases touch no lock
  and no shared cache line. Same API shape, `'static` initializers.

```rust
use par_buffer_pool::BufferPool;
use rayon::prelude::*;

// one pool per scratch kind, built once before the loop
let pool = BufferPool::new(|| vec![0.0f64; 1 << 20]);

let totals: Vec<f64> = (0..1000)
    .into_par_iter()
    .map(|task| {
        let mut buf = pool.get(); // fresh on first lease, recycled after
        buf.fill(task as f64);
        buf.iter().sum::<f64>()   // buf returns itself at scope end
    })
    .collect();
```

- **No `put` to forget.** `get` returns a *guard* (`SharedPooled` /
  `LocalPooled`), not a bare value; dropping the guard — normally, on an
  early return, or during a panic unwind — returns the buffer to the pool
  automatically.
- **Self-limiting memory.** The pool allocates only when the relevant slot
  is momentarily empty, so buffer count converges to peak concurrent leases
  (≈ worker count), instead of one big transient allocation per task.
- **`Clone` handles.** Cheaply clone a pool into worker closures or store it
  in a driver struct; both are shared-state handles, like `Arc`.
- **Zero dependencies, zero `unsafe`.** One `Mutex<Vec<T>>`, or one
  `thread_local!` registry — a closure and two atomics each.

It is the polished form of the pool snippet that tends to get hand-copied
into parallel codebases — the manual `get`/`put` dance around every closure,
with the returns forgotten on some path.

## The problem

Parallel code loves this shape:

```rust
(0..ntasks).into_par_iter().for_each(|task| {
    let mut scratch = vec![0u8; 1 << 20]; // fresh malloc per task
    let mut header = String::new();       // ...and another
    // a few dozen microseconds of work, then both are dropped
});
```

Every task pays allocation + zero-fill for buffers that the *next* task will
immediately need again, the allocator juggles `nthreads` big transient chunks,
and memory footprint spikes per task instead of per thread. The usual
hand-rolled fix is a manual pool:

```rust
let mut scratch = scratch_pool.get(); // ...
// every early exit needs its own matching put —
// real code has been seen with four put sites in one closure
scratch_pool.put(scratch);
```

which shifts the burden onto every caller, every branch, every panic path.
This crate is that idea with the failure modes removed: the checkout *is* a
scope guard.

## Which pool?

Both pools share one API shape — `new` / `get` (a guard) / `with` / `put` /
`into_inner` / `with_reset` / `stats` — so switching is mostly a type swap.
The difference is the lease mechanism, and it shows up in exactly one place:
**many small tasks at high worker counts favor `ThreadLocalPool`; everything
else is a feature choice.**

| | [`BufferPool`] | [`ThreadLocalPool`] |
|---|---|---|
| Storage | one shared `Mutex<Vec<T>>` | one slot per worker thread (`thread_local!`) |
| Lease cost | mutex pair ≈ 26 ns, contended under many tiny tasks | TLS access ≈ 9 ns, never contended |
| Initializer | may borrow non-`'static` data (`BufferPool<'a, T>`) | must be `'static` |
| Buffers per pool | ≈ peak concurrent leases | ≈ worker threads, even if only two are busy |
| Buffer movement | returns to the shared pile from any thread | parks on the thread that drops the guard; migrates if that is not where it was leased |
| After the pool is dropped | buffers freed with the pool | reclaimed on each thread's next pool interaction, or at thread exit (never unbounded) |
| Idle cap | `with_max_idle` | not needed: each thread parks at most one buffer |
| `idle_len` | global parked count | this thread's parked count (0 or 1) |

(Lease costs measured on a 16-core desktop CPU with glibc, default features;
see `examples/local_static_bench.rs` for the full matrix and the caveat that
orderings should be re-measured on target hardware.)

## API tour

| Item | What it does |
|---|---|
| `Pool::new(init)` | Build a pool; `init` runs lazily, only when a lease finds the slot/pile empty. |
| `Pool::get()` | Lease a buffer as a guard (`SharedPooled<T>` / `LocalPooled<T>`, Deref/DerefMut to `T`). |
| guard drop | Return the buffer — on scope exit, early return, or panic unwind. |
| `Pool::with(f)` | Closure-based checkout of the same mechanism. |
| `guard.into_inner()` | Detach the buffer when it *is* the result (not recycled). |
| `Pool::put(buf)` | Manual return for detached/raw buffers. |
| `Pool::with_reset(f)` | Run `f(&mut buf)` on every return, so leases start in a known state (e.g. zeroed). |
| `BufferPool::with_max_idle(n)` | Cap idle buffers; returns beyond the cap are dropped. (`ThreadLocalPool` needs no cap.) |
| `Pool::stats()` | `leases` / `allocations` counters — proof the pool works, and the number for memory budgeting. Behind the `stats` feature (off by default). |

The pools never clear buffers on their own — accumulation buffers should use
`with_reset(|b| b.fill(0.0))`, buffers fully overwritten by every task should
skip it.

`BufferPool<'a, T>` is `Send + Sync` whenever `T: Send`; `SharedPooled<'a, T>`
is `Send` under the same condition, so leases may even migrate between
threads. `ThreadLocalPool<T>` is always `Send + Sync`; its guard is `Send`
when `T: Send`.

## Examples

The [`examples/`](examples) directory is the documentation of record; each is
runnable (`cargo run --release --example <name>`):

| Example | Pattern |
|---|---|
| [`rayon_scratch`] | The canonical one: a scratch pool feeding a rayon loop (hex-encoding binary blobs), stats printout. |
| [`pair_scores`] | All-pairs tasks, one `O(n²)` scratch matrix per pair; shows churn drop from `O(ntasks)` buffers to a near-constant pooled count. |
| [`detach_collect`] | Scratch vs. result in the same task: Mandelbrot strips detach via `into_inner` into the image, escape-time scratch recycles. |
| [`size_buckets`] | Variable-size workloads: a grow-only pool (`clear` + `resize` per lease) and power-of-two size-class pools. |
| [`scoped_threads`] | No rayon: `std::thread::scope`, cloned handles, an initializer borrowing stack-local config, and a reset hook. |
| [`alloc_bench`] | `fresh` vs `pooled` vs rayon `map_init` on 2 MiB buffers: wall time (best-of-N) plus allocation counts. |
| [`local_static_bench`] | The two pools head-to-head against `fresh`/`map_init`/a hoisted floor: lease-cost microbench plus parallel tiny/huge task regimes. |

## Feature flags

- **`stats`** (off by default): per-pool lease/allocation counters via
  `stats()` — `leases`, `allocations`, and `reuses()`. Costs one relaxed
  atomic add per lease on a shared cache line (for both pools); enable it to
  verify that recycling is happening or to budget scratch memory. The
  crate's own tests and examples enable it automatically via a
  dev-dependency on the crate itself, so `cargo test` and
  `cargo run --example ...` need no extra flags.

## Design notes

- **`BufferPool` locking.** One `Mutex` guards a `Vec`; the critical section
  is `pop`/`push` (~tens of ns). New buffers are allocated outside the lock.
  Contention only matters for low-microsecond tasks at high worker counts —
  then coarsen tasks or use `ThreadLocalPool`.
- **`ThreadLocalPool` mechanics.** One concrete `thread_local!` registry per
  process maps dense pool ids to slots; a lease is a TLS access plus an
  `Option::take` of a type-erased buffer (`Box<dyn Any + Send>`, downcast on
  checkout — no `unsafe`). When a pool is dropped it flips a shared liveness
  flag and bumps a global epoch; each thread's next registry access sweeps
  its own dead slots (one `Acquire` load on the fast path), and thread exit
  drops the whole registry — so parked buffers are reclaimed lazily but
  never leak past the thread's life.
- **Why not raw `thread_local!` + `RefCell`?** That is the classic
  alternative — and `ThreadLocalPool` keeps its access cost while adding the
  missing structure: a guard (no borrowed-flag panics across APIs), reset
  hooks, detach-into-result, stats, reclamation after the pool is dropped,
  and usage across library boundaries without exposing TLS internals.
- **Why not rayon `for_each_init`/`map_init`?** Fine — and lock-free — when a
  *single* loop owns all its scratch; but scratch cannot detach into results,
  and the initializer re-runs per loop invocation, so loops called repeatedly
  (per request, per frame, per batch) keep churning. Pools also serve plain
  threads and nested parallelism.
- **Poisoning.** The `BufferPool` lock never guards an invariant (just
  values), so a poisoned mutex is recovered via `PoisonError::into_inner` —
  no panic loops, no leaked idle buffers.
- **Memory budgeting.** With the `stats` feature, `stats().allocations`
  bounds concurrently outstanding leases: budget scratch as
  `allocations × buffer_len × size_of::<T>()`. `with_max_idle` keeps bursts
  from parking `BufferPool` buffers forever; `ThreadLocalPool` is bounded by
  construction.
- **Honest limits.** Recycled buffers carry stale contents (use
  `with_reset`); a recycled small hot buffer from the shared pool pays a
  cross-core cache-line transfer on its next use; tiny cheap buffers are
  better left to the allocator.

## Testing

`cargo test` covers the guard semantics of both pools (scope exit, early
return, panic unwind, cross-thread drop/migration), detach/re-pool, reset
hooks, idle caps, handle cloning, nested leases, reclamation of dropped
pools' parked buffers, non-`'static` initializers under
`std::thread::scope`, and rayon stress tests driving 20 000 leases through
both pools.

## License

MIT OR Apache-2.0, at your option (license files to be added on first
publication).

[`BufferPool`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html
[`ThreadLocalPool`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.ThreadLocalPool.html
[`rayon_scratch`]: examples/rayon_scratch.rs
[`pair_scores`]: examples/pair_scores.rs
[`detach_collect`]: examples/detach_collect.rs
[`size_buckets`]: examples/size_buckets.rs
[`scoped_threads`]: examples/scoped_threads.rs
[`alloc_bench`]: examples/alloc_bench.rs
[`local_static_bench`]: examples/local_static_bench.rs
