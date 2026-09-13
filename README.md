# par-buffer-pool

A tiny, dependency-free, thread-safe **buffer pool** with **RAII guards**, for
reusing scratch buffers across parallel workers — rayon, scoped threads, or a
single thread.

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

- **No `put` to forget.** [`BufferPool::get`] returns a *guard*, not a bare
  value; dropping the guard — normally, on an early return, or during a panic
  unwind — returns the buffer to the pool automatically.
- **Self-limiting memory.** The pool allocates only when momentarily empty, so
  buffer count converges to peak concurrent leases (≈ worker count), instead
  of one big transient allocation per task.
- **`Clone` handle, non-`'static` initializer.** Cheaply clone the pool into
  worker closures or store it in a driver struct; the initializer may borrow
  dimensions, config, anything local from the caller — no `'static`, no
  cloning, no leaking.
- **Zero dependencies, zero `unsafe`.** One `Mutex<Vec<T>>`, a closure, two
  atomics. `#![forbid(unsafe_code)]`.

It is the polished form of the `BufferPool` snippet that tends to get
hand-copied into parallel codebases — the manual `get`/`put` dance around
every closure, with the returns forgotten on some path.

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
`par-buffer-pool` is that idea with the failure modes removed: the checkout
*is* a scope guard.

## API tour

| Item | What it does |
|---|---|
| [`BufferPool::new(init)`] | Build a pool; `init` runs lazily, only when a lease finds the pool empty. |
| [`BufferPool::get()`] | Lease a buffer as a `Pooled<T>` guard (Deref/DerefMut to `T`). |
| `Pooled` drop | Return the buffer — on scope exit, early return, or panic unwind. |
| [`Pooled::into_inner()`] | Detach the buffer when it *is* the result (not recycled). |
| [`BufferPool::put(buf)`] | Manual return for detached/raw buffers. |
| [`BufferPool::with_reset(f)`] | Run `f(&mut buf)` on every return, so leases start in a known state (e.g. zeroed). |
| [`BufferPool::with_max_idle(n)`] | Cap idle buffers; returns beyond the cap are dropped. |
| [`BufferPool::stats()`] | `leases` / `allocations` counters — proof the pool works, and the number for memory budgeting. |
| `BufferPool: Clone` | Cheap shared-state handle (like `Arc`); guards keep the pool alive. |

The pool never clears buffers on its own — accumulation buffers should use
`with_reset(|b| b.fill(0.0))`, buffers fully overwritten by every task should
skip it.

`BufferPool<'a, T>` is `Send + Sync` whenever `T: Send`; `Pooled<'a, T>` is
`Send` under the same condition, so leases may even migrate between threads.

## Examples

The [`examples/`](examples) directory is the documentation of record; each is
runnable (`cargo run --release --example <name>`):

| Example | Pattern |
|---|---|
| [`rayon_scratch`] | The canonical one: a scratch pool feeding a rayon loop (hex-encoding binary blobs), stats printout. |
| [`pair_scores`] | All-pairs tasks, one `O(n²)` scratch matrix per pair; shows churn drop from `O(ntasks)` buffers to `O(nthreads)`. |
| [`detach_collect`] | Scratch vs. result in the same task: Mandelbrot strips detach via `into_inner` into the image, escape-time scratch recycles. |
| [`size_buckets`] | Variable-size workloads: a grow-only pool (`clear` + `resize` per lease) and power-of-two size-class pools. |
| [`scoped_threads`] | No rayon: `std::thread::scope`, cloned handles, an initializer borrowing stack-local config, and a reset hook. |
| [`alloc_bench`] | `fresh` vs `pooled` vs rayon `map_init` on 2 MiB buffers: wall time (best-of-N) plus allocation counts. |

## Design notes

- **Why not `thread_local!`?** It is the classic alternative and the classic
  trap: generic thread-local scratch that borrows non-`'static` data is
  famously awkward, and thread-locals persist for the life of the thread. A
  shared pool has none of those problems and dies with its last handle.
- **Why not per-thread slots (`Vec<Mutex<T>>` indexed by thread id)?** Fragile
  sizing, idle buffers for absent threads, and no sharing across differently
  shaped loops.
- **Why not rayon `for_each_init`/`map_init`?** Fine — and lock-free — when a
  *single* loop owns all its scratch; but scratch cannot detach into results,
  and the initializer re-runs per loop invocation, so loops called repeatedly
  (per request, per frame, per batch) keep churning. Pools also serve plain
  threads and nested parallelism.
- **Locking.** One `Mutex` guards a `Vec`; the critical section is `pop`/`push`
  (~tens of ns). New buffers are allocated outside the lock. Contention only
  matters for low-microsecond tasks — then coarsen tasks or use `map_init`.
- **Poisoning.** The lock never guards an invariant (just values), so a
  poisoned mutex is recovered via `PoisonError::into_inner` — no panic loops,
  no leaked idle buffers.
- **Memory budgeting.** `stats().allocations` bounds concurrently outstanding
  leases: budget scratch as `allocations × buffer_len × size_of::<T>()`.
  `with_max_idle` keeps bursts from parking buffers forever.
- **Honest limits.** Recycled buffers carry stale contents (use `with_reset`);
  a recycled small hot buffer pays a cross-core cache-line transfer on its
  next use; tiny cheap buffers are better left to the allocator.

## Testing

`cargo test` covers the guard semantics (scope exit, early return, panic
unwind, cross-thread drop), detach/re-pool, reset hooks, idle caps, handle
cloning, non-`'static` initializers under `std::thread::scope`, and a rayon
stress test asserting `allocations ≤ nthreads` across 20 000 leases.

## License

MIT OR Apache-2.0, at your option (license files to be added on first
publication).

[`BufferPool::new(init)`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.new
[`BufferPool::get()`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.get
[`BufferPool::put(buf)`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.put
[`BufferPool::with_reset(f)`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.with_reset
[`BufferPool::with_max_idle(n)`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.with_max_idle
[`BufferPool::stats()`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html#method.stats
[`Pooled::into_inner()`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.Pooled.html#method.into_inner
[`rayon_scratch`]: examples/rayon_scratch.rs
[`pair_scores`]: examples/pair_scores.rs
[`detach_collect`]: examples/detach_collect.rs
[`size_buckets`]: examples/size_buckets.rs
[`scoped_threads`]: examples/scoped_threads.rs
[`alloc_bench`]: examples/alloc_bench.rs
