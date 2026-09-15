# Comparison with other pool crates

Buffer pools are a well-trodden corner of crates.io. This page surveys the
landscape honestly: what existing crates do, what this crate does and does
not offer, and where the differences actually matter.

**Scope and method.** All facts below (versions, features, release status)
were checked against docs.rs and crates.io on **2026-09-14**; versions drift,
so re-verify before relying on a specific cell. "✗" means the feature is
absent, "n/d" means the documentation does not settle it — that is not a
flaw, just something this page refuses to guess at. This page compares
*features*, not benchmarks: lease costs depend on hardware, allocator, and
task shape, and the crate docs point at the local benchmarks for that.

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
- **Both flavors, one dependency-free crate.** A shared mutex pile and a
  thread-local slot pool with the same guard API — with `forbid(unsafe_code)`
  — is a combination no other surveyed crate offers (opool comes closest and
  pays with a crossbeam dependency; the zero-dep crates have one flavor or
  are single-threaded).
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

## Choosing, briefly

If your scratch buffers are small or your loop is a single `for_each`, use
nothing: `vec![]` or rayon's `map_init`/`for_each_init` (see the crate docs'
["When *not* to use this"](crate#when-not-to-use-this)). If you want RAII
scratch reuse in a thread-safe, zero-dependency, `#![forbid(unsafe_code)]`
package — with borrowed initializers, an optional reset hook, idle cap,
prefill, drain, and stats — this crate is built for exactly that. If you
need `no_std`, return-validity checks, or prefer an allocator trait,
[opool] is the strongest alternative.
