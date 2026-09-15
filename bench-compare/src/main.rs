//! Cross-crate buffer-pool benchmark (`bench-compare`).
//!
//! Backs the "Measured lease costs" section of the crate's
//! `src/comparison.md`; `output.txt` is the logged run those tables quote.
//! The crate is excluded from the published package (`exclude` in the
//! root `Cargo.toml`): it is a benchmark harness, not part of the
//! library. Published-crate rows only (no prototypes — the sharded design
//! was adopted in round 3).
//!
//! Fairness rules, applied to every row of every scenario:
//!   * same buffer type and size per scenario (`Vec<u8>`), same `init`
//!     closure for every pool;
//!   * no reset hooks anywhere — except swimmer, whose `Recyclable` impl
//!     for `Vec` clears the buffer on return, forcing one `resize` per
//!     lease (counted, documented);
//!   * steady state: a warm-up pass runs before any timing, so rows measure
//!     the recycle path, not first-touch allocation;
//!   * opaque per-lease work: `touch`/`stream` are `#[inline(never)]` and
//!     derive addresses from a `black_box`ed index, so LLVM cannot fold the
//!     loop into a closed form (a first draft without this "measured" a
//!     1 KiB malloc+zero at 1.1 ns);
//!   * best-of-5 wall clock per row (`Instant`), identical rep counts;
//!   * capped pools all get the same cap (64): object-pool at construction,
//!     opool via `new_prefilled(64, ..)`; the rest are uncapped;
//!   * `par-buffer-pool` is used with default features (`stats` OFF), like
//!     a downstream user.

use std::hint::black_box;
use std::thread;
use std::time::Instant;

use lockfree_object_pool::LinearObjectPool;
use object_pool::Pool as ObjectPool;
use opool::Pool as Opool;
use par_buffer_pool::{BufferPool, ThreadLocalPool};
use rayon::prelude::*;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const REPS: usize = 5;
/// Rough single-core boost clock of the test machine (AMD Ryzen 9 9950X3D,
/// `performance` governor); only used for the convenience "≈ cycles" column
/// of the single-threaded rows. Multi-core rows report ns only — all-core
/// clocks differ and a cycles column would overstate precision there.
const BOOST_GHZ: f64 = 5.7;

fn buf(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

/// Near-zero work: three cache lines at varying, opaque addresses.
#[inline(never)]
fn touch(buf: &mut [u8], i: usize) -> u8 {
    let len = buf.len();
    let i = black_box(i);
    let a = i % len;
    let b = (i * 7) % len;
    let c = len - 1 - (i % 3);
    buf[a] = buf[a].wrapping_add(1);
    buf[b] = buf[b].wrapping_add(2);
    buf[c] = buf[c].wrapping_add(3);
    buf[a].wrapping_add(buf[b]).wrapping_add(buf[c])
}

/// One write+read pass over every cache line: the per-task scratch traffic
/// of a realistic task (the buffer is fully used, not merely touched).
#[inline(never)]
fn stream(buf: &mut [u8], i: usize) -> u64 {
    let i = black_box(i) as u8;
    let mut acc = 0u64;
    for chunk in buf.chunks_mut(64) {
        chunk[0] = chunk[0].wrapping_add(i);
        acc += chunk[0] as u64;
    }
    acc
}

// The hand-rolled floor: plain thread_local! + RefCell, no structure.
thread_local! {
    static RAW: std::cell::RefCell<Option<Vec<u8>>> = const { std::cell::RefCell::new(None) };
}

fn raw_get(n: usize) -> Vec<u8> {
    RAW.with(|c| c.borrow_mut().take())
        .unwrap_or_else(|| buf(n))
}

fn raw_put(v: Vec<u8>) {
    RAW.with(|c| *c.borrow_mut() = Some(v));
}

/// Best-of-REPS serial harness. `f` is monomorphized: the timed loop calls
/// it with zero dispatch overhead, identically for every row.
fn bench_serial(name: &str, iters: usize, warm: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..warm {
        f(i);
    }
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        for i in 0..iters {
            f(i);
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("{name:>28}: {:>7.2} ns/lease", best * 1e9 / iters as f64);
    best
}

/// Best-of-REPS rayon harness: one lease (or malloc) per task.
fn bench_rayon(name: &str, tasks: usize, f: impl Fn(usize) + Sync + Send) -> f64 {
    (0..tasks / 4).into_par_iter().for_each(&f); // warm-up to steady state
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        (0..tasks).into_par_iter().for_each(&f);
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "{name:>28}: {:>7.1} ms  ({:>6.1} ns/task)",
        best * 1e3,
        best * 1e9 / tasks as f64
    );
    best
}

/// Best-of-REPS `std::thread::scope` harness: fresh threads each rep, one
/// lease per iteration per thread, same code shape as the rayon scenarios.
fn bench_scope(name: &str, threads: usize, iters: usize, f: impl Fn(usize) + Sync + Send) -> f64 {
    let total = threads * iters;
    let warm = || {
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| (0..iters / 4).for_each(&f));
            }
        })
    };
    warm();
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t0 = Instant::now();
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| (0..iters).for_each(&f));
            }
        });
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "{name:>28}: {:>7.1} ms  ({:>6.1} ns/task)",
        best * 1e3,
        best * 1e9 / total as f64
    );
    best
}

fn cycles(ns: f64) -> f64 {
    ns * BOOST_GHZ
}

/// Serial lease+drop with near-zero work: the pure mechanism cost.
fn serial_lease(n: usize, iters: usize) {
    println!(
        "\n=== S: serial lease, {} leases of {} KiB Vec<u8>, near-zero work ===",
        iters,
        n / KIB
    );

    let fresh = bench_serial("fresh vec![0u8; N]", iters, iters / 10, |i| {
        let mut b = buf(n);
        black_box(touch(&mut b, i));
    });
    let hoisted = {
        let mut b = buf(n);
        bench_serial("hoisted local (floor)", iters, iters / 10, |i| {
            black_box(touch(&mut b, i));
        })
    };
    let raw = bench_serial("raw thread_local!", iters, iters / 10, |i| {
        let mut b = raw_get(n);
        black_box(touch(&mut b, i));
        raw_put(b);
    });

    let local_pool = ThreadLocalPool::new(move || buf(n));
    let local = {
        let p = &local_pool;
        bench_serial("ours ThreadLocalPool", iters, iters / 10, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let shared_pool = BufferPool::new(move || buf(n));
    let shared = {
        let p = &shared_pool;
        bench_serial("ours BufferPool (get)", iters, iters / 10, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let owned = {
        let p = &shared_pool;
        bench_serial("ours BufferPool (get_owned)", iters, iters / 10, |i| {
            let mut b = p.get_owned();
            black_box(touch(&mut b, i));
        })
    };

    let object = ObjectPool::new(64, move || buf(n));
    let object = {
        let p = &object;
        bench_serial("object-pool 0.6", iters, iters / 10, |i| {
            let mut b = p.pull(move || buf(n));
            black_box(touch(&mut b, i));
        })
    };

    struct Alloc {
        n: usize,
    }
    impl opool::PoolAllocator<Vec<u8>> for Alloc {
        fn allocate(&self) -> Vec<u8> {
            buf(self.n)
        }
    }
    let alloc = Alloc { n };
    let opool_c = Opool::new_prefilled(64, alloc);
    let opool_c = {
        let p = &opool_c;
        bench_serial("opool 0.2 Pool", iters, iters / 10, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let opool_l = opool::LocalPool::new_prefilled(64, Alloc { n });
    let opool_l = {
        let p = &opool_l;
        bench_serial("opool 0.2 LocalPool", iters, iters / 10, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let swim: swimmer::Pool<Vec<u8>> = swimmer::builder().with_supplier(move || buf(n)).build();
    // swimmer's `Recyclable for Vec` clears on return: the resize is part of
    // its per-lease contract for `Vec` buffers.
    let swim = {
        let p = &swim;
        bench_serial("swimmer 0.3 (+resize)", iters, iters / 10, |i| {
            let mut b = p.get();
            b.resize(n, 0);
            black_box(touch(&mut b, i));
        })
    };

    let lockfree = LinearObjectPool::new(move || buf(n), |_: &mut Vec<u8>| {});
    let lockfree = {
        let p = &lockfree;
        bench_serial("lockfree-object-pool 0.1", iters, iters / 10, |i| {
            let mut b = p.pull();
            black_box(touch(&mut b, i));
        })
    };

    let rows = [
        ("fresh", fresh),
        ("hoisted", hoisted),
        ("raw TLS", raw),
        ("ours local", local),
        ("ours shared", shared),
        ("ours shared owned", owned),
        ("object-pool", object),
        ("opool Pool", opool_c),
        ("opool LocalPool", opool_l),
        ("swimmer", swim),
        ("lockfree", lockfree),
    ];
    println!(
        "{:>28}   (≈ cycles at {BOOST_GHZ} GHz)",
        format!("{:>28}:", "name: ns/lease")
    );
    for (name, secs) in rows {
        let ns = secs * 1e9 / iters as f64;
        println!("{name:>28}: {:>7.2} ns  ≈ {:>4.0} cycles", ns, cycles(ns));
    }
}

/// Rayon scenario with near-zero work: contention / scheduling regime.
fn rayon_tiny(n: usize, tasks: usize) {
    println!(
        "\n=== P1: rayon, {tasks} tiny tasks, {} KiB Vec<u8>, near-zero work ===",
        n / KIB
    );

    let fresh = bench_rayon("fresh vec![0u8; N]", tasks, |i| {
        let mut b = buf(n);
        black_box(touch(&mut b, i));
    });

    let map_init = {
        (0..tasks / 4)
            .into_par_iter()
            .map_init(|| buf(n), |b, j: usize| black_box(touch(b, j)))
            .for_each(|_| {});
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t0 = Instant::now();
            (0..tasks)
                .into_par_iter()
                .map_init(|| buf(n), |b, j: usize| black_box(touch(b, j)))
                .for_each(|_| {});
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "{:>28}: {:>7.1} ms  ({:>6.1} ns/task)",
            "rayon map_init (floor)",
            best * 1e3,
            best * 1e9 / tasks as f64
        );
        best
    };

    let local_pool = ThreadLocalPool::new(move || buf(n));
    let local = {
        let p = &local_pool;
        bench_rayon("ours ThreadLocalPool", tasks, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let shared_pool = BufferPool::new(move || buf(n));
    let shared = {
        let p = &shared_pool;
        bench_rayon("ours BufferPool", tasks, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let object = ObjectPool::new(64, move || buf(n));
    let object = {
        let p = &object;
        bench_rayon("object-pool 0.6", tasks, |i| {
            let mut b = p.pull(move || buf(n));
            black_box(touch(&mut b, i));
        })
    };

    struct Alloc {
        n: usize,
    }
    impl opool::PoolAllocator<Vec<u8>> for Alloc {
        fn allocate(&self) -> Vec<u8> {
            buf(self.n)
        }
    }
    let opool_c = Opool::new_prefilled(64, Alloc { n });
    let opool = {
        let p = &opool_c;
        bench_rayon("opool 0.2 Pool", tasks, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let swim: swimmer::Pool<Vec<u8>> = swimmer::builder().with_supplier(move || buf(n)).build();
    let swim = {
        let p = &swim;
        bench_rayon("swimmer 0.3 (+resize)", tasks, |i| {
            let mut b = p.get();
            b.resize(n, 0);
            black_box(touch(&mut b, i));
        })
    };

    let lockfree = LinearObjectPool::new(move || buf(n), |_: &mut Vec<u8>| {});
    let lockfree = {
        let p = &lockfree;
        bench_rayon("lockfree-object-pool 0.1", tasks, |i| {
            let mut b = p.pull();
            black_box(touch(&mut b, i));
        })
    };

    let ns = |secs: f64| secs * 1e9 / tasks as f64;
    println!("  ns/task over the map_init floor:");
    for (name, secs) in [
        ("fresh", fresh),
        ("ours local", local),
        ("ours shared", shared),
        ("object-pool", object),
        ("opool Pool", opool),
        ("swimmer", swim),
        ("lockfree", lockfree),
    ] {
        println!("{name:>28}: +{:.1}", ns(secs - map_init));
    }
}

/// Rayon scenario with realistic per-task work: every cache line of the
/// buffer written+read once per task (~µs tasks). The common shape the
/// crate is built for.
fn rayon_realistic(n: usize, tasks: usize, label: &str) {
    println!(
        "\n=== P2: rayon, {tasks} tasks of {label}, {} KiB Vec<u8>, full per-task pass ===",
        n / KIB
    );

    let fresh = bench_rayon("fresh vec![0u8; N]", tasks, |i| {
        let mut b = buf(n);
        black_box(stream(&mut b, i));
    });

    let map_init = {
        (0..tasks / 4)
            .into_par_iter()
            .map_init(|| buf(n), |b, j: usize| black_box(stream(b, j)))
            .for_each(|_| {});
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t0 = Instant::now();
            (0..tasks)
                .into_par_iter()
                .map_init(|| buf(n), |b, j: usize| black_box(stream(b, j)))
                .for_each(|_| {});
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "{:>28}: {:>7.1} ms  ({:>6.1} ns/task)",
            "rayon map_init (floor)",
            best * 1e3,
            best * 1e9 / tasks as f64
        );
        best
    };

    let local_pool = ThreadLocalPool::new(move || buf(n));
    let local = {
        let p = &local_pool;
        bench_rayon("ours ThreadLocalPool", tasks, |i| {
            let mut b = p.get();
            black_box(stream(&mut b, i));
        })
    };

    let shared_pool = BufferPool::new(move || buf(n));
    let shared = {
        let p = &shared_pool;
        bench_rayon("ours BufferPool", tasks, |i| {
            let mut b = p.get();
            black_box(stream(&mut b, i));
        })
    };

    let object = ObjectPool::new(64, move || buf(n));
    let object = {
        let p = &object;
        bench_rayon("object-pool 0.6", tasks, |i| {
            let mut b = p.pull(move || buf(n));
            black_box(stream(&mut b, i));
        })
    };

    struct Alloc {
        n: usize,
    }
    impl opool::PoolAllocator<Vec<u8>> for Alloc {
        fn allocate(&self) -> Vec<u8> {
            buf(self.n)
        }
    }
    let opool_c = Opool::new_prefilled(64, Alloc { n });
    let opool = {
        let p = &opool_c;
        bench_rayon("opool 0.2 Pool", tasks, |i| {
            let mut b = p.get();
            black_box(stream(&mut b, i));
        })
    };

    let swim: swimmer::Pool<Vec<u8>> = swimmer::builder().with_supplier(move || buf(n)).build();
    let swim = {
        let p = &swim;
        bench_rayon("swimmer 0.3 (+resize)", tasks, |i| {
            let mut b = p.get();
            b.resize(n, 0);
            black_box(stream(&mut b, i));
        })
    };

    let lockfree = LinearObjectPool::new(move || buf(n), |_: &mut Vec<u8>| {});
    let lockfree = {
        let p = &lockfree;
        bench_rayon("lockfree-object-pool 0.1", tasks, |i| {
            let mut b = p.pull();
            black_box(stream(&mut b, i));
        })
    };

    let ns = |secs: f64| secs * 1e9 / tasks as f64;
    println!("  ns/task over the map_init floor:");
    for (name, secs) in [
        ("fresh", fresh),
        ("ours local", local),
        ("ours shared", shared),
        ("object-pool", object),
        ("opool Pool", opool),
        ("swimmer", swim),
        ("lockfree", lockfree),
    ] {
        println!("{name:>28}: +{:.1}", ns(secs - map_init));
    }
}

/// No-rayon scenario: `std::thread::scope`, the other documented driver.
fn scope_plain(n: usize, threads: usize, iters: usize) {
    println!(
        "\n=== T: std::thread::scope, {threads} threads x {iters} leases, {} KiB Vec<u8>, near-zero work ===",
        n / KIB
    );

    let fresh = bench_scope("fresh vec![0u8; N]", threads, iters, |i| {
        let mut b = buf(n);
        black_box(touch(&mut b, i));
    });

    let raw = bench_scope("raw thread_local!", threads, iters, |i| {
        let mut b = raw_get(n);
        black_box(touch(&mut b, i));
        raw_put(b);
    });

    let local_pool = ThreadLocalPool::new(move || buf(n));
    let local = {
        let p = &local_pool;
        bench_scope("ours ThreadLocalPool", threads, iters, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let shared_pool = BufferPool::new(move || buf(n));
    let shared = {
        let p = &shared_pool;
        bench_scope("ours BufferPool", threads, iters, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let object = ObjectPool::new(64, move || buf(n));
    let object = {
        let p = &object;
        bench_scope("object-pool 0.6", threads, iters, |i| {
            let mut b = p.pull(move || buf(n));
            black_box(touch(&mut b, i));
        })
    };

    struct Alloc {
        n: usize,
    }
    impl opool::PoolAllocator<Vec<u8>> for Alloc {
        fn allocate(&self) -> Vec<u8> {
            buf(self.n)
        }
    }
    let opool_c = Opool::new_prefilled(64, Alloc { n });
    let opool = {
        let p = &opool_c;
        bench_scope("opool 0.2 Pool", threads, iters, |i| {
            let mut b = p.get();
            black_box(touch(&mut b, i));
        })
    };

    let swim: swimmer::Pool<Vec<u8>> = swimmer::builder().with_supplier(move || buf(n)).build();
    let swim = {
        let p = &swim;
        bench_scope("swimmer 0.3 (+resize)", threads, iters, |i| {
            let mut b = p.get();
            b.resize(n, 0);
            black_box(touch(&mut b, i));
        })
    };

    let lockfree = LinearObjectPool::new(move || buf(n), |_: &mut Vec<u8>| {});
    let lockfree = {
        let p = &lockfree;
        bench_scope("lockfree-object-pool 0.1", threads, iters, |i| {
            let mut b = p.pull();
            black_box(touch(&mut b, i));
        })
    };

    let ns = |secs: f64| secs * 1e9 / (threads * iters) as f64;
    println!("  ns/task vs raw thread_local! floor:");
    for (name, secs) in [
        ("fresh", fresh),
        ("ours local", local),
        ("ours shared", shared),
        ("object-pool", object),
        ("opool Pool", opool),
        ("swimmer", swim),
        ("lockfree", lockfree),
    ] {
        println!("{name:>28}: +{:.1}", ns(secs - raw));
    }
}

fn main() {
    // Explicit global pool: the bench must not inherit an ambient
    // RAYON_NUM_THREADS (this shell had one set), and a default downstream
    // user gets `available_parallelism` workers.
    let par = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(par)
        .build_global()
        .expect("build rayon global pool");
    println!(
        "bench-compare: available_parallelism = {par}, rayon workers = {}, best of {REPS}",
        rayon::current_num_threads()
    );

    serial_lease(KIB, 2_000_000);
    serial_lease(64 * KIB, 300_000);
    rayon_tiny(KIB, 1_000_000);
    rayon_realistic(64 * KIB, 200_000, "~µs work each");
    rayon_realistic(2 * MIB, 16_384, "~100s of µs work each");
    scope_plain(KIB, par, 1_000_000);
}
