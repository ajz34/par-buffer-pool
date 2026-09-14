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
`into_inner` / `with_reset` / `stats`, plus `with_max_idle` and `drain` on
[`BufferPool`] only — so switching is mostly a type swap.
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
| `drain` | the whole idle pile at once (call it between phases: leases still out are silently not included) | none: other threads' slots are unreachable (pool drop / thread exit reclaims) |

(Lease costs measured on a 16-core desktop CPU with glibc, default features;
see `examples/local_static_bench.rs` for the full matrix and the caveat that
orderings should be re-measured on target hardware.)

## Semantics

The pools never clear buffers on their own — accumulation buffers should use
`with_reset(|b| b.fill(0.0))`, buffers fully overwritten by every task should
skip it.

`BufferPool<'a, T>` is `Send + Sync` whenever `T: Send`; `SharedPooled<'a, T>`
is `Send` under the same condition, so leases may even migrate between
threads. `ThreadLocalPool<T>` is always `Send + Sync`; its guard is `Send`
when `T: Send`.

The full API tour — every method, one line each — lives on the
[docs.rs page](https://docs.rs/par-buffer-pool).

## Examples

The [`examples/`](examples) directory is the documentation of record; each is
runnable (`cargo run --release --example <name>`):

| Example | Pattern |
|---|---|
| [`rayon_scratch`] | The canonical one: a scratch pool feeding a rayon loop (hex-encoding binary blobs), stats printout. |
| [`drain_reduce`] | Reduction without `fold`: pooled `f64` accumulators collect partial sums in a plain parallel `for_each`; `drain` hands them back for the final combine. |
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

## Documentation

The [docs.rs page](https://docs.rs/par-buffer-pool) is the comprehensive
reference: the API tour, per-item documentation with scraped usage examples
from [`examples/`](examples), design notes, and the testing overview all
live there. The docs.rs build also renders the private modules, so the
internals are reviewable without cloning the source.

## License

Dual-licensed, at your option: the MIT License or the Apache License,
Version 2.0 (`license = "MIT OR Apache-2.0"` in `Cargo.toml`). Full license
texts are added to the repository on first publication.

## Provenance

Most of this crate — code, tests, examples, and documentation — was written
by AI coding agents (GLM-5.3 and GLM-5.3-flash) at the direction of the
repository maintainer.

[`BufferPool`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.BufferPool.html
[`ThreadLocalPool`]: https://docs.rs/par-buffer-pool/latest/par_buffer_pool/struct.ThreadLocalPool.html
[`rayon_scratch`]: examples/rayon_scratch.rs
[`drain_reduce`]: examples/drain_reduce.rs
[`pair_scores`]: examples/pair_scores.rs
[`detach_collect`]: examples/detach_collect.rs
[`size_buckets`]: examples/size_buckets.rs
[`scoped_threads`]: examples/scoped_threads.rs
[`alloc_bench`]: examples/alloc_bench.rs
[`local_static_bench`]: examples/local_static_bench.rs
