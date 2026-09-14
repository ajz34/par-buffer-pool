//! Correctness tests for `ThreadLocalPool`: RAII return semantics, detach,
//! reset hook, thread-scoped `idle_len`, nested leases, reclamation after
//! drop, buffer migration, handle cloning, and behavior under rayon.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use par_buffer_pool::{LocalPooled, ThreadLocalPool};
use rayon::prelude::*;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn handle_and_guard_are_send_and_sync() {
    assert_send::<ThreadLocalPool<Vec<f64>>>();
    assert_sync::<ThreadLocalPool<Vec<f64>>>();
    assert_send::<LocalPooled<'static, Vec<f64>>>();
}

#[test]
fn drop_parks_buffer_in_this_threads_slot() {
    // 'static initializer: count through an Arc (BufferPool could borrow the
    // counter instead — this bound is the documented difference).
    let init_runs = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&init_runs);
    let pool = ThreadLocalPool::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        vec![0u8; 16]
    });

    {
        let mut buf = pool.get();
        buf[0] = 42; // contents survive the round trip
    }

    assert_eq!(pool.idle_len(), 1, "parked on this thread");
    let recycled = pool.get();
    assert_eq!(recycled[0], 42, "the very buffer we dropped came back");
    assert_eq!(init_runs.load(Ordering::SeqCst), 1, "no second allocation");
    assert_eq!(
        pool.stats(),
        par_buffer_pool::PoolStats {
            leases: 2,
            allocations: 1
        }
    );
}

#[test]
fn idle_len_is_thread_scoped() {
    // The divergence from BufferPool: each thread's slot is invisible to
    // other threads, so the main thread sees 0 even while a worker parks.
    let pool = ThreadLocalPool::new(|| vec![0u32; 8]);
    thread::scope(|s| {
        s.spawn(|| {
            drop(pool.get());
            assert_eq!(pool.idle_len(), 1, "visible on the parking thread");
        });
    });
    assert_eq!(
        pool.idle_len(),
        0,
        "another thread's parked buffer is not ours to count"
    );
}

#[test]
fn early_return_parks_buffer() {
    fn task(pool: &ThreadLocalPool<Vec<i32>>, n: i32) -> i32 {
        let mut buf = pool.get();
        if n < 0 {
            return -1; // early return must not lose the buffer
        }
        buf.push(n);
        buf.iter().sum()
    }

    let pool = ThreadLocalPool::new(Vec::new);
    assert_eq!(task(&pool, -5), -1);
    assert_eq!(task(&pool, 3), 3);
    assert_eq!(pool.idle_len(), 1);
}

#[test]
fn panic_unwind_parks_buffer() {
    let pool = ThreadLocalPool::new(String::new);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut buf = pool.get();
        buf.push_str("dirty");
        panic!("task failed mid-lease");
    }));
    assert!(result.is_err());
    assert_eq!(
        pool.idle_len(),
        1,
        "guard dropped during unwind parks the buffer"
    );
}

#[test]
fn into_inner_detaches_and_put_re_pools() {
    let pool = ThreadLocalPool::new(Vec::<u8>::new);

    let owned = {
        let mut buf = pool.get();
        buf.extend_from_slice(b"result");
        buf.into_inner() // keep it
    };
    assert_eq!(owned, b"result");
    assert_eq!(pool.idle_len(), 0);

    // a detached buffer can be re-pooled by hand once it is done being a result
    pool.put(owned);
    assert_eq!(pool.idle_len(), 1);
    assert!(pool.get().starts_with(b"result"));
}

#[test]
fn reset_hook_runs_on_every_return() {
    let pool = ThreadLocalPool::new(|| vec![1.0f64; 4]).with_reset(|buf| buf.fill(0.0));

    let mut first = pool.get();
    first.fill(9.0);
    drop(first); // reset via guard drop
    assert!(
        pool.get().iter().all(|&x| x == 0.0),
        "leased clean after drop-return"
    );

    let mut second = pool.get();
    second.fill(8.0);
    let raw = second.into_inner();
    pool.put(raw); // reset via manual put
    assert!(
        pool.get().iter().all(|&x| x == 0.0),
        "leased clean after put-return"
    );
}

#[test]
fn with_checks_out_and_returns_like_a_guard() {
    let pool = ThreadLocalPool::new(|| vec![0u8; 8]);
    let got = pool.with(|buf| {
        buf[0] = 7;
        buf[0]
    });
    assert_eq!(got, 7);
    assert_eq!(pool.idle_len(), 1);
    assert_eq!(pool.stats().allocations, 1);
}

#[test]
fn nested_leases_allocate_fresh_instead_of_panicking() {
    // Take-based checkouts: a nested get/with while a buffer is checked out
    // sees an empty slot and allocates, exactly like leasing twice from
    // BufferPool. The single slot means the outer park replaces the inner
    // one — a slot holds at most one buffer per pool.
    let pool = ThreadLocalPool::new(|| vec![0u8; 4]);
    pool.with(|outer| {
        pool.with(|inner| {
            outer[0] = 1;
            inner[0] = 2;
        });
        assert_eq!(pool.idle_len(), 1, "inner parked; outer still checked out");
        assert_eq!(outer[0], 1);
    });
    assert_eq!(pool.idle_len(), 1);
    let buf = pool.get();
    assert_eq!(buf[0], 1, "the outer buffer is the parked one");
    drop(buf);
    assert_eq!(pool.stats().allocations, 2, "two buffers were created");
}

#[test]
fn sequential_leases_recycle_one_buffer() {
    let pool = ThreadLocalPool::new(|| vec![0u32; 64]);
    for k in 0..100 {
        pool.get()[0] = k; // lease, touch, drop — all in one statement
    }
    let stats = pool.stats();
    assert_eq!(stats.leases, 100);
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.reuses(), 99);
}

#[test]
fn guard_migrates_to_whichever_thread_drops_it() {
    // A lease taken on a worker thread can finish on the main thread; the
    // buffer then parks in the *main* thread's slot (no global pile).
    let pool = ThreadLocalPool::new(|| vec![0u32; 8]);
    let buf = thread::scope(|s| {
        s.spawn(|| {
            let mut buf = pool.get();
            buf[0] = 7;
            buf // send the guard back mid-lease
        })
        .join()
        .unwrap()
    });
    assert_eq!(buf[0], 7);
    drop(buf); // parked on the *main* thread now
    assert_eq!(
        pool.idle_len(),
        1,
        "the buffer migrated to the dropping thread's slot"
    );
}

#[test]
fn dropping_the_pool_reclaims_parked_buffers_on_other_threads() {
    struct Tracked(Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    let pool = {
        let counter = Arc::clone(&drops);
        ThreadLocalPool::new(move || Tracked(Arc::clone(&counter)))
    };

    thread::scope(|s| {
        s.spawn(|| {
            drop(pool.get()); // parked on the worker thread
        });
    });
    assert_eq!(drops.load(Ordering::SeqCst), 0, "parked, pool still alive");
    drop(pool); // the worker's parked buffer is now unreachable except via TLS

    // Any thread's next registry access sweeps its own dead slots.
    let other = ThreadLocalPool::new(|| ());
    let _ = other.idle_len();
    thread::scope(|s| {
        s.spawn(|| {
            let _ = other.idle_len(); // the worker's sweep
        });
    });
    assert_eq!(
        drops.load(Ordering::SeqCst),
        1,
        "worker reclaimed the dead pool's buffer"
    );
}

#[test]
fn cloned_handles_share_state() {
    let pool = ThreadLocalPool::new(|| vec![0u16; 4]);
    let clone = pool.clone();

    let buf = pool.get();
    assert_eq!(clone.idle_len(), 0);
    drop(buf);
    assert_eq!(clone.idle_len(), 1, "state is shared, not copied");

    assert_eq!(clone.stats().leases, 1);
}

#[test]
fn rayon_stress_allocates_at_most_one_buffer_per_thread() {
    let ntasks = 20_000;
    let pool = ThreadLocalPool::new(|| vec![0u64; 64]);
    let checksum: u64 = (0..ntasks)
        .into_par_iter()
        .map(|task| {
            let mut buf = pool.get();
            buf.fill(task as u64);
            buf.iter().sum::<u64>()
        })
        .sum();

    assert!(checksum > 0);
    let stats = pool.stats();
    let nthreads = rayon::current_num_threads();
    assert_eq!(stats.leases, ntasks);
    assert!(stats.allocations >= 1, "work actually happened");
    assert!(
        stats.allocations <= nthreads,
        "allocations ({}) exceeded thread count ({})",
        stats.allocations,
        nthreads
    );
    assert!(
        pool.idle_len() <= 1,
        "this thread parks at most one buffer, however busy the workers were"
    );
}

#[test]
fn static_initializer_with_scoped_threads() {
    // The initializer must be 'static (unlike BufferPool's): `move` the
    // dimensions in.
    let dims = [32_usize, 48];
    let pool = ThreadLocalPool::new(move || vec![0.0f64; dims[0] * dims[1]]);

    thread::scope(|s| {
        for t in 0..4 {
            let pool = pool.clone();
            s.spawn(move || {
                let mut buf = pool.get();
                assert_eq!(buf.len(), 32 * 48);
                buf[0] = t as f64;
            });
        }
    });

    assert_eq!(pool.stats().leases, 4);
    assert!(pool.stats().allocations <= 4);
}

#[test]
fn debug_impls_do_not_panic() {
    let pool = ThreadLocalPool::new(|| vec![1.0f64; 2]);
    let guard = pool.get();
    let _ = format!("{pool:?} {guard:?} {:?}", pool.stats());
}

#[test]
fn pooled_works_for_non_default_and_odd_types() {
    // T needs no Default/Clone/Debug — only the initializer's output.
    struct Opaque(#[allow(dead_code)] u64);
    let pool = ThreadLocalPool::new(|| Opaque(3));
    let _ = pool.get();
    let _ = pool.get();
    assert_eq!(pool.stats().allocations, 1);
}

#[test]
fn pools_of_many_types_share_one_registry_safely() {
    // One concrete TLS registry serves every T; ids must not collide.
    let a = ThreadLocalPool::new(|| vec![1u8; 3]);
    let b = ThreadLocalPool::new(|| String::from("b"));
    let c = ThreadLocalPool::new(|| (1u32, 2u32));

    assert_eq!(&*a.get(), &[1, 1, 1]);
    assert_eq!(&*b.get(), "b");
    assert_eq!(&*c.get(), &(1, 2));
    drop(a.get());
    drop(b.get());
    drop(c.get());
    assert_eq!(a.idle_len(), 1);
    assert_eq!(b.idle_len(), 1);
    assert_eq!(c.idle_len(), 1);
    assert_eq!(&*a.get(), &[1, 1, 1], "no cross-type slot confusion");
}
