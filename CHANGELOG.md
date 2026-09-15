# Changelog

## v0.1.0 -- 2026-09-15

Initial release of par-buffer-pool: a tiny, dependency-free,
`#![forbid(unsafe_code)]` crate with two thread-safe buffer pools with RAII
guards, for reusing scratch buffers across parallel workers (rayon, scoped
threads, ...).

New features:

- `BufferPool`, a shared, sharded pool of reusable buffers; any worker thread
  can satisfy any lease. The idle pile is sharded, so leasing stays cheap when
  many workers lease and return buffers concurrently. `get` returns a
  `SharedPooled` guard instead of a bare buffer, so the buffer returns to the
  pool on drop — also on early return or panic unwind. There is no `put` to
  forget.
- `ThreadLocalPool`, the lock-free sibling: one buffer slot per worker thread;
  leases touch no lock and no shared cache line. Guards are `LocalPooled`.
- Initializers that borrow: `BufferPool`'s buffer initializer and reset hook
  are bounded by the pool's lifetime `'a` instead of `'static`, so plain stack
  locals (a dimensions tuple, a per-run preset) can be captured without `move`,
  `Clone`, or `Arc`.
- Pool conveniences: `Default` construction, `prefill` to fill the pool up
  front, and `drain` to empty the idle storage into a `Vec`.
- Optional `stats` feature (off by default): per-pool lease/allocation
  counters, to verify recycling behavior or budget scratch memory.

Documentation update:

- docs.rs is configured to document all features and private items, and to
  scrape the `examples/` call sites into the per-item API docs; the examples
  double as usage documentation.
- The measured cross-crate comparison (object-pool, opool, swimmer,
  lockfree-object-pool) lives in the `comparison` module; its benchmark
  harness is a separate crate excluded from the published package.
