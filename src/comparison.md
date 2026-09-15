# Comparison with other pool crates

Buffer pools are a well-trodden corner of crates.io. This page surveys the
landscape honestly: what existing crates offer, what this crate does and
does not offer — in features, and in measured lease cost. The benchmark
section at the bottom states its full machine, software, and per-scenario
conditions next to the results, so the numbers can be interpreted — or
re-measured — somewhere else.

**Scope and method.** Feature facts (versions, capabilities, release
status) were checked against docs.rs and crates.io on **2026-09-15**;
versions drift, so re-verify before relying on a specific cell. "✗" means
the feature is absent, "n/d" means the documentation does not settle it —
that is not a flaw, just something this page refuses to guess at. Measured
numbers are reported in nanoseconds with a rough CPU-cycle conversion;
they describe one machine (fully identified below) and will differ on
yours — the *orderings* are the portable part, and even those deserve a
re-measure before an important decision.

## What this crate is

[`BufferPool`](crate::BufferPool) and [`ThreadLocalPool`](crate::ThreadLocalPool)
pool *whole, reusable scratch values* (usually buffers) with an RAII checkout:
`get` returns a guard ([`SharedPooled`](crate::SharedPooled) /
[`LocalPooled`](crate::LocalPooled)) that returns the buffer on drop. There is
no `put` to forget, buffers are built by a user closure, and the shared pool's
closure may borrow non-`'static` data. That is the whole idea — it is not an
allocator, not an arena, and not a connection pool.

## The direct comparables

Crates whose core purpose is the same: hand out reusable objects and take
them back.

| Crate | Shared pool | Thread-local pool | Guarded checkout | Reset on return | Borrowing (`'a`) init | Idle cap | Prefill | Runtime deps | Last release / status |
|---|---|---|---|---|---|---|---|---|---|
| **this crate** | ✓ sharded mutex pile | ✓ `ThreadLocalPool` | ✓ `Deref` guards, `into_inner`, `put` | ✓ closure hook | ✓ `'a`-bounded closure | ✓ `with_max_idle` | ✓ `BufferPool::prefill` | **zero**, `#![forbid(unsafe_code)]` | this release |
| [object-pool] 0.6.0 | ✓ (`Arc`-shared) | ✗ | ✓ `Reusable`, owned variant, `detach`/`attach` | **✗** (docs warn objects are "returned but NOT reset") | ✗ no lifetime parameter | ✓ cap at construction, `try_pull` saturates | ✗ | `parking_lot` | 2024-08, slow-moving |
| [opool] 0.2.0 | ✓ lock-free (`ArrayQueue`) | ✓ `LocalPool` | ✓ `RefGuard`/`RcGuard`, owned variants | ✓ via `PoolAllocator::reset` (+ `is_valid` rejection) | ✗ allocator trait, no lifetime parameter | ✓ `pool_size` caps what is stored | ✓ `new_prefilled` | `crossbeam-queue`; `no_std` + alloc | 2025-12, active |
| [buffer-pool (quiche)] 0.2.1 | ✓ sharded | sharded (not exposed per-thread) | ✓ `Pooled` guard | ✓ `Reuse` trait, may reject the return | n/d (thin docs) | n/d | n/d | `crossbeam` + `foundations` | 2026-02, active inside quiche |
| [swimmer] 0.3.0 | ✓ | ✓ per-thread sharded | ✓ `Recycled` | ✓ `Recyclable::recycle` trait (no custom hook) | n/d | n/d | ✓ starting size | `thread_local` 0.3 (+ optional) | 2021, dormant |
| [lifeguard] 0.6.1 | **✗ `Rc`-based, single-threaded** | ✗ | ✓ `Recycled` / `RcRecycled` | ✓ `Recycleable` trait | n/d | ✓ `MaxSize` setting | ✓ `StartingSize` | **zero** | 2020, dormant |
| [lockfree-object-pool] 0.1.6 | ✓ linear / spin / mutex variants | ✗ | ✓ guards + owned variants | ✓ second closure of `new(init, reset)` | n/d | n/d | ✗ | **zero** | 2024-05, slow |
| [syncpool] 0.1.6 | ✓ | ✗ | **✗ explicit `get()`/`put()`** | ✗ | n/d | ✗ grows on starvation | ✓ `with_size` | zero | 2021, dormant |
| [bufferpool] 0.1.7 | slices of *one* contiguous allocation | n/d | ✓ RAII reference | n/a (regions returned) | n/d | fixed by builder | ✓ via builder | n/d | 2026-06, small |

[object-pool]: https://docs.rs/object-pool
[opool]: https://docs.rs/opool
[buffer-pool (quiche)]: https://docs.rs/buffer-pool
[swimmer]: https://docs.rs/swimmer
[lifeguard]: https://docs.rs/lifeguard
[lockfree-object-pool]: https://docs.rs/lockfree-object-pool
[syncpool]: https://docs.rs/syncpool
[bufferpool]: https://docs.rs/bufferpool

Notes on the closest neighbors:

- **object-pool** is the most direct rival and the most widely used. It
  popularized the guard model this crate also uses. It has no reset hook —
  its own docs warn that objects come back dirty — no stats, no thread-local
  flavor, and its capacity is fixed at construction (beyond the cap,
  `try_pull` refuses and `pull` allocates a fresh object, where this crate's
  `with_max_idle` silently drops *returns* and `get` always succeeds).
- **opool** is the most feature-complete neighbor: both pool flavors, a
  reset hook, a *validity check on return* (neat: `is_valid` decides whether
  an object may be re-pooled at all), prefill, and `no_std`. The trade-offs
  are a `crossbeam-queue` dependency, allocator-trait ceremony instead of a
  plain closure, and no borrowed (non-`'static`) initializers.
- **buffer-pool** (Cloudflare, inside quiche) is the nearest in spirit:
  sharded, drop-based, with a return policy (`Reuse` decides repool vs.
  drop). It is built for quiche's needs, minimally documented outside its
  repo, and pulls two non-trivial dependencies — a different point on the
  same design space, oriented at byte buffers for QUIC.
- **swimmer** and **lifeguard** are the previous generation (trait-based
  `Recyclable` recycling, guard types). lifeguard is single-threaded
  (`Rc` in its API) but zero-dependency; swimmer is per-thread sharded,
  conceptually the closest published analog to
  [`ThreadLocalPool`](crate::ThreadLocalPool). Both have been dormant for
  years.
- **lockfree-object-pool** takes closures for both init *and* reset (like
  this crate), ships several backend implementations, and is
  zero-dependency; its capacity/overflow policy is undocumented, and there
  are no stats or drain.
- **syncpool** is the counter-example: an explicit `get()`/`put()` pool —
  precisely the "no put to forget" failure mode this crate is built around.
- **bufferpool** (bennetthardwick) is a different design point: sub-slicing
  one contiguous allocation into non-overlapping regions, rather than
  recycling whole buffers.

## Adjacent, but different problems

| Category | Crates | Why it is not this |
|---|---|---|
| Arena / bump allocation | [`bumpalo`], [`typed-arena`] | Fast allocation, but no per-object free: everything dies with the arena. Pools recycle *individual* buffers across phases. |
| Ref-counted byte buffers | [`bytes`] (`BytesMut`/`Bytes`) | Solves zero-copy *sharing/slicing*, not scratch reuse; bytes-only. Compatible: view wrappers over a pooled `Vec` work fine (see the crate docs' views section). |
| Index-based slot storage | [`slab`] | Pre-allocated keyed storage; no checkout/return, no RAII. |
| Fixed-capacity queues | [`crossbeam-queue`] `ArrayQueue` | The building block other pools use, not a pool API (no init, no guards, no reset). |
| Embedded static pools | [`heapless::pool`], [`embedded-buffer-pool`] | `no_std` pools over compile-time-fixed memory for MCUs; a different runtime model than `std` thread pools. |
| Connection pools | [`r2d2`], [`deadpool`], [`bb8`] | Pool *connections* (health checks, await-queues, async managers); nothing to do with scratch buffers. |
| Recycled allocations | [`recycle_vec`] | Recycles one `Vec`'s allocation into a differently-typed `Vec`; not a pool. |

[`bumpalo`]: https://docs.rs/bumpalo
[`typed-arena`]: https://docs.rs/typed-arena
[`bytes`]: https://docs.rs/bytes
[`slab`]: https://docs.rs/slab
[`crossbeam-queue`]: https://docs.rs/crossbeam-queue
[`heapless::pool`]: https://docs.rs/heapless/latest/heapless/pool/index.html
[`embedded-buffer-pool`]: https://crates.io/crates/embedded-buffer-pool
[`r2d2`]: https://docs.rs/r2d2
[`deadpool`]: https://docs.rs/deadpool
[`bb8`]: https://docs.rs/bb8
[`recycle_vec`]: https://crates.io/crates/recycle_vec

Deprecated or miscategorized names that search engines surface, for the
record: [`mempool`] (BurntSushi) is self-declared "UNMAINTAINED AND
DEPRECATED", and is a *lending* pool (references never leave the pool);
`poolite` is a thread pool, not a buffer pool; carllerche's `pool` has been
abandoned since 2017.

[`mempool`]: https://docs.rs/mempool

## What this crate has that the others do not

- **Initializers (and reset hooks) that borrow.** `BufferPool<'a, T>` bounds
  its closure by the pool's lifetime, so per-run configuration can be
  captured by reference — no `'static`, no `move`, no `Arc`, no `Clone`.
  None of the surveyed crates document a lifetime-parameterized pool; it is
  this crate's founding feature.
- **Both flavors, one dependency-free crate.** A shared sharded pool and a
  thread-local slot pool with the same guard API — with
  `forbid(unsafe_code)` — is a combination no other surveyed crate offers
  (opool comes closest and pays with a crossbeam dependency; the zero-dep
  crates have one flavor or are single-threaded).
- **`drain`.** Taking the whole idle pile back as owned buffers (a reduction
  combine step, a memory reclaim) is not offered by any surveyed crate.
- **`stats`.** Lease/allocation counters behind an off-by-default feature
  (proof recycling happens, and a scratch-memory budgeting number). No
  surveyed pool crate exposes counters.
- **Documented reclamation story for the thread-local pool.** Parked buffers
  survive their pool's drop safely and are reclaimed on the next
  interaction or at thread exit — swimmer's sharded design documents no
  such contract.

## What the others have that this crate does not

Honesty requires the reverse list:

- **Validity check on return.** opool's `is_valid` and quiche's `Reuse ->
  bool` can reject a worn-out buffer at return time. This crate has no hook
  for that (emulable at the call site: `into_inner()` a buffer you do not
  want back, then drop it).
- **`no_std` support.** opool works on `no_std` + alloc; this crate is
  `std`-only (`std::sync::Mutex`, `thread_local!`).
- **`Rc`-flavored guards** for single-threaded pools (opool, lifeguard).
  Mostly moot here: `get`'s lease path already takes no reference count
  (the guard borrows the pool), and a non-`Send` `T` works with
  `BufferPool` single-threaded.
- **Allocator-trait configuration** (opool) instead of closures — a style
  difference that buys `no_std` and per-object `is_valid`, at the cost of
  more ceremony for the 95% case.
- **Async checkout.** No surveyed *scratch* pool has await-based checkout
  (the async crates in this space are connection pools). This crate's guards
  are runtime-agnostic — they work inside async tasks — but holding a guard
  across an `.await` blocks that buffer for the whole suspension, same as
  with every crate above.

## Measured lease costs

Pooling is a mechanism-cost claim as much as a feature set, so this page
also carries measurements — with the conditions stated beside them, because
a lease-cost number without its machine, allocator, and workload is not a
fact. All rows below come from one purpose-written harness,
[`bench-compare`](https://github.com/ajz34/par-buffer-pool/tree/main/bench-compare)
in the repository (with its logged run in `bench-compare/output.txt`),
which prices every crate under identical rules: the same buffer type and
size per scenario, the same init closure, no reset hooks for anyone
(except swimmer, whose `Recyclable` impl for `Vec` forces one `resize` per
lease — counted in its rows), identical caps (64) for the capped crates, a
warm-up pass before every timing, opaque per-lease work that LLVM cannot
fold away, and best-of-5 wall clock. `par-buffer-pool` is used with
default features (`stats` off), like a downstream user.

### Benchmark conditions

| Condition | Value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D — 16 cores / 32 hardware threads, 1 socket / 1 NUMA node, max boost ≈ 5.7 GHz, `performance` governor |
| RAM | 64 GiB DDR5 |
| OS | Linux 7.0.0-31-generic, glibc malloc |
| Rust | rustc 1.97.1, release profile, `stats` feature off |
| Parallelism | rayon global pool = `available_parallelism()` = 32 workers; scope scenario = 32 `std::thread` threads |
| Buffer | `Vec<u8>` — 1 KiB, 64 KiB, or 2 MiB per scenario (below); identical init closure for every pool |
| Per-task work | *near-zero*: 3 cache lines at `black_box`ed addresses; *full pass*: one write+read over every cache line of the buffer |
| Timing | best of 5 wall-clock repetitions (`std::time::Instant`) after a warm-up pass |
| Competitors | object-pool 0.6.0, opool 0.2.0, swimmer 0.3.0, lockfree-object-pool 0.1.6, rayon 1.12.0 (all latest as of the date above) |

Scenarios:

| Scenario | Driver | Workers | Buffer | Per-task work | Tasks | Metric |
|---|---|---|---|---|---|---|
| S1 | single thread | 1 | 1 KiB | near-zero | 2 000 000 leases | ns/lease |
| S2 | single thread | 1 | 64 KiB | near-zero | 300 000 leases | ns/lease |
| P1 | rayon | 32 | 1 KiB | near-zero | 1 000 000 | ns/task |
| P2a | rayon | 32 | 64 KiB | full pass (≈ µs) | 200 000 | ns/task |
| P2b | rayon | 32 | 2 MiB | full pass (≈ 100s of µs) | 16 384 | ns/task |
| T | `thread::scope` | 32 | 1 KiB | near-zero | 32 × 1 000 000 | ns/task |

### S1/S2 — single thread, lease + drop (ns/lease, best of 5)

Rows sorted fastest first; the "≈ cycles" column divides by the 5.7 GHz
boost clock — a convenience for intuition, not a hardware guarantee.

| Pool | S1 (1 KiB) | ≈ cycles | S2 (64 KiB) | ≈ cycles |
|---|---|---|---|---|
| hoisted stack local (floor) | 2.2 | 13 | 2.2 | 13 |
| opool `LocalPool` (single-owner) | 3.5 | 20 | 3.8 | 21 |
| raw `thread_local!` + `RefCell` | 7.0 | 40 | 7.3 | 42 |
| lockfree-object-pool 0.1 | 9.8 | 56 | 10.4 | 59 |
| **ours `ThreadLocalPool`** | **10.3** | **59** | **10.3** | **58** |
| opool `Pool` (concurrent) | 10.8 | 61 | 10.8 | 62 |
| swimmer 0.3 (+ forced resize) | 12.6 | 72 | **215.1** | **1 226** |
| object-pool 0.6 | 16.1 | 92 | 16.2 | 92 |
| **ours `BufferPool`** | **23.3** | **133** | **23.4** | **133** |
| **ours `BufferPool::get_owned`** | **30.7** | **175** | **30.3** | **173** |
| fresh `vec![0u8; N]` | 26.9 | 153 | **241.2** | **1 375** |

Honest readings:

- `ThreadLocalPool` costs ~3 ns over raw TLS (the registry's epoch check and
  dense-id lookup pay for safe reclamation after pool drop) and is in the
  same band as the other lock-free pools. opool's `LocalPool` is faster
  still, but it is a different capability class: a single-owner deque that
  cannot be shared with worker threads at all.
- `BufferPool`'s uncontended lease (~130 cycles) is the price of
  *cross-thread satisfiability*: a shard lock pair plus the guard's
  `Option` plumbing, where the TLS pool only touches thread-local storage.
  That is the trade the table in
  ["Which pool"](crate#which-pool) describes.
- `get_owned` shows exactly what the docs charge it for: +~7 ns (≈ 40
  cycles) of `Arc` reference-count traffic per lease.
- `fresh` grows with buffer size (64 KiB ≈ 240 ns ≈ 1 400 cycles per
  malloc+zero); pooled rows do not. swimmer's 64 KiB row includes its
  forced `clear()` + `resize`, which re-touches the whole buffer.

### P1/P2 — rayon, 32 workers (ns/task, best of 5)

`map_init` is rayon's own per-worker scratch; it is the "no pool needed"
floor for a single loop. Deltas are against it.

| Pool | P1 tiny, 1 KiB | Δ vs floor | P2a full pass, 64 KiB | Δ vs floor | P2b full pass, 2 MiB | Δ vs floor |
|---|---|---|---|---|---|---|
| rayon `map_init` (floor) | 0.3 | — | 31.0 | — | 2 192 | — |
| **ours `ThreadLocalPool`** | **0.8** | **+0.6** | **29.5** | **−1.4** | **1 497** | **−696** |
| **ours `BufferPool`** | **1.0** | **+0.7** | **29.5** | **−1.5** | **1 522** | **−670** |
| swimmer 0.3 (+ resize) | 1.1 | +0.8 | 45.6 | +14.6 | 3 296 | +1 103 |
| fresh `vec![0u8; N]` | 1.4 | +1.2 | 46.2 | +15.2 | 3 134 | +942 |
| lockfree-object-pool 0.1 | 43.0 | +42.8 | 80.7 | +49.7 | 1 893 | −300 |
| opool `Pool` (CAS queue) | 254.1 | +253.8 | 260.8 | +229.8 | 21 838 | +19 646 |
| object-pool 0.6 (mutex pile) | 1 551.4 | +1 551.1 | 1 893.1 | +1 862.2 | 1 836 | −357 |

Honest readings:

- This is the regime the [choosing guide](crate#which-pool) points at, now
  measured: at 32 workers with tiny tasks, every *per-thread* design
  (`ThreadLocalPool`, swimmer's shards) and this crate's sharded
  `BufferPool` sit within a nanosecond of the floor, while every *single
  synchronized structure* pays hundreds to thousands of ns/task — the
  crates' own mechanisms become the workload. The sharding is what moved
  `BufferPool` from the convoyed group into the floor group.
- With real per-task work (P2a, ≈ µs), the two "ours" rows are
  indistinguishable from the floor — lease cost vanishes, which is why the
  choosing guide says bigger tasks make the pool choice a feature choice.
  They even land ~1.5 ns/task *below* `map_init` (noise at this scale, but
  consistent across runs).
- P2b (2 MiB) is memory-dominated, and the floor label inverts: `map_init`
  rebuilds its scratch on every call, so each timed call re-pays allocation
  *and first-touch page faults* for 32 × 2 MiB, while a pool that lives
  across the whole run keeps its already-faulted buffers. That is a genuine
  pooling advantage (allocation avoidance includes fault avoidance), not a
  rounding artifact — but it also means this row measures scratch lifetime,
  not lease mechanism.
- The opool P2b row is the same effect, amplified by its FIFO queue: with
  64 buffers cycling through 32 workers, the hot working set is every
  buffer (128 MiB, exceeding L3) instead of one per worker, so the CAS
  queue's cost is dwarfed by cache-miss traffic. object-pool's LIFO pile
  keeps buffer-to-worker affinity and stays at the floor group. FIFO vs
  LIFO recycling order is a real design axis for large buffers.
- `fresh` at 1 KiB is nearly free here because glibc serves repeated
  same-size mallocs from per-thread caches — pooling's win at small sizes
  is footprint and churn (`stats().allocations` vs one allocation per
  task), not wall clock.

### T — `std::thread::scope`, 32 threads (ns/task, best of 5)

Same shape as P1 without rayon: 32 scoped threads × 1 000 000 leases each
of 1 KiB, near-zero work; wall time includes thread spawn (identically for
every row). Delta is against the raw `thread_local!` floor.

| Pool | ns/task | Δ vs raw TLS |
|---|---|---|
| raw `thread_local!` + `RefCell` (floor) | 0.3 | — |
| **ours `ThreadLocalPool`** | **0.9** | **+0.6** |
| **ours `BufferPool`** | **1.0** | **+0.7** |
| swimmer 0.3 (+ resize) | 1.0 | +0.7 |
| fresh `vec![0u8; N]` | 1.5 | +1.2 |
| lockfree-object-pool 0.1 | 43.1 | +42.8 |
| opool `Pool` (CAS queue) | 254.7 | +254.4 |
| object-pool 0.6 (mutex pile) | 1 513.4 | +1 513.1 |

The rayon and scoped-thread tables agree: which driver you use matters far
less than whether the pool serializes its tenants.

### How to read all of this

- **Nanoseconds are local.** These rows describe one desktop CPU, one
  allocator, one rayon configuration. The orderings (per-thread vs shared
  structures, floor vs convoy, pool vs fresh by size) are the transferable
  part; re-measure on your hardware with your buffer shapes before making
  a decision that matters. The in-repo `local_static_bench` example
  reproduces the two-pool part of this on your machine.
- **What is *not* measured:** cold-start behavior (a pool's first leases
  run the initializer), `drain`/`prefill`/idle-cap bookkeeping (they are
  off the lease path by design), reset-hook cost (no crate in the table
  uses one except swimmer's forced `Vec` reset), and allocation counts
  (the `stats` feature reports those at runtime).
- The winner in the contended tables is often "no pool at all" —
  `map_init`, or plain `vec![]` for large cheap buffers. This crate's docs
  open with
  [when *not* to use it](crate#when-not-to-use-this), and the measurements
  back that up: for a single `for_each` loop with per-worker scratch,
  rayon's built-in is the right tool. What a pool adds is the same guard
  pattern across loops, threads, and phases — with reset, detach, drain,
  idle caps, and stats attached.

## Choosing, briefly

If your scratch buffers are small or your loop is a single `for_each`, use
nothing: `vec![]` or rayon's `map_init`/`for_each_init` (see the crate docs'
["When *not* to use this"](crate#when-not-to-use-this)). If you want RAII
scratch reuse in a thread-safe, zero-dependency, `#![forbid(unsafe_code)]`
package — with borrowed initializers, an optional reset hook, idle cap,
prefill, drain, and stats — this crate is built for exactly that. If you
need `no_std`, return-validity checks, or prefer an allocator trait,
[opool] is the strongest alternative.
