//! Correctness tests for `par_buffer_pool`: RAII return semantics, detach,
//! reset hook, idle cap, handle cloning, non-`'static` initializers, and
//! behavior under rayon.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use par_buffer_pool::{BufferPool, SharedPooled};
use rayon::prelude::*;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn send_sync_when_t_is_send() {
    assert_send::<BufferPool<'static, Vec<f64>>>();
    assert_sync::<BufferPool<'static, Vec<f64>>>();
    assert_send::<SharedPooled<'static, Vec<f64>>>();
    // non-Send T is still usable single-threaded; the pool just is not Sync
    let rc_pool = BufferPool::new(|| Rc::new(7u32));
    assert_eq!(*rc_pool.get(), Rc::new(7u32));
}

#[test]
fn drop_returns_buffer_to_pool() {
    let init_runs = AtomicUsize::new(0);
    let pool = BufferPool::new(|| {
        init_runs.fetch_add(1, Ordering::SeqCst);
        vec![0u8; 16]
    });

    {
        let mut buf = pool.get();
        buf[0] = 42; // contents survive the round trip
    }

    assert_eq!(pool.idle_len(), 1);
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
fn early_return_returns_buffer() {
    fn task(pool: &BufferPool<Vec<i32>>, n: i32) -> i32 {
        let mut buf = pool.get();
        if n < 0 {
            return -1; // early return must not lose the buffer
        }
        buf.push(n);
        buf.iter().sum()
    }

    let pool = BufferPool::new(Vec::new);
    assert_eq!(task(&pool, -5), -1);
    assert_eq!(task(&pool, 3), 3);
    assert_eq!(pool.idle_len(), 1);
}

#[test]
fn panic_unwind_returns_buffer() {
    let pool = BufferPool::new(String::new);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut buf = pool.get();
        buf.push_str("dirty");
        panic!("task failed mid-lease");
    }));
    assert!(result.is_err());
    assert_eq!(
        pool.idle_len(),
        1,
        "guard dropped during unwind recycles the buffer"
    );
}

#[test]
fn into_inner_detaches_and_put_re_pools() {
    let pool = BufferPool::new(Vec::<u8>::new);

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
    let pool = BufferPool::new(|| vec![1.0f64; 4]).with_reset(|buf| buf.fill(0.0));

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
fn max_idle_caps_parked_buffers() {
    let pool = BufferPool::new(Vec::<u8>::new).with_max_idle(3);
    for i in 0..10u8 {
        pool.put(vec![i]);
    }
    assert_eq!(pool.idle_len(), 3, "excess buffers are dropped, not parked");
    // LIFO: the last parked buffer (pushes 3..=9 were dropped at the cap).
    // The cap must not affect leasing: a full pool still hands buffers out.
    // (Bind the guard: a temporary in `assert_eq!` would drop only at the
    // end of the statement, after the check below runs.)
    let parked = pool.get();
    assert_eq!(*parked, vec![2]);
    drop(parked);
    // the lease left 2 idle; returning it refilled to 3 (still under cap)
    assert_eq!(pool.idle_len(), 3);
}

#[test]
fn drain_empties_the_pool_into_a_vec() {
    // Hold three leases at once and return them: sequential get/drop cycles
    // would just recycle one buffer (it parks and is re-leased immediately).
    let pool = BufferPool::new(|| vec![0u8; 8]);
    let holders: Vec<_> = (0..3)
        .map(|i| {
            let mut buf = pool.get();
            buf[0] = i as u8;
            buf
        })
        .collect();
    drop(holders); // three buffers parked, in this drop order
    assert_eq!(pool.idle_len(), 3);
    let stats_before = pool.stats();

    let drained: Vec<Vec<u8>> = pool.drain();
    assert_eq!(
        drained.iter().map(|b| b[0]).collect::<Vec<_>>(),
        vec![0, 1, 2],
        "drain yields the parked buffers, in parking order"
    );
    assert_eq!(pool.idle_len(), 0, "the pool is empty afterwards");
    assert_eq!(pool.drain().len(), 0, "draining an empty pool is fine");
    assert_eq!(
        pool.stats(),
        stats_before,
        "drain is not a lease/allocation"
    );

    // still a working pool: the next lease finds it empty and allocates
    let fresh = pool.get();
    assert_eq!(fresh[0], 0, "a freshly initialized buffer");
    drop(fresh);
    assert_eq!(pool.stats().allocations, stats_before.allocations + 1);
}

#[test]
fn drain_skips_outstanding_leases_and_the_reset_hook() {
    // The reset hook borrows a stack local: drain must leave it alone.
    let resets = AtomicUsize::new(0);
    let pool = BufferPool::new(|| vec![1.0f64; 4]).with_reset(|buf| {
        buf.fill(0.0);
        resets.fetch_add(1, Ordering::SeqCst);
    });

    let held = pool.get(); // outstanding for the whole test (not drainable)
    let mut returned = pool.get(); // a second buffer, parked mid-test
    returned.fill(8.0);
    drop(returned); // return path: hook #1
    let resets_after_return = resets.load(Ordering::SeqCst);

    let drained = pool.drain();
    assert_eq!(drained.len(), 1, "the outstanding lease is not in the pile");
    assert_eq!(
        drained[0],
        vec![0.0; 4],
        "parked after its return-time reset"
    );
    assert_eq!(
        resets.load(Ordering::SeqCst),
        resets_after_return,
        "drain takes buffers out; the return-path hook does not run"
    );
    assert_eq!(pool.idle_len(), 0);

    drop(held); // parks after the drain, through the hook as usual
    assert_eq!(pool.idle_len(), 1);
    assert_eq!(resets.load(Ordering::SeqCst), resets_after_return + 1);
}

#[test]
fn cloned_handles_share_state() {
    let pool = BufferPool::new(|| vec![0u16; 4]);
    let clone = pool.clone();

    let buf = pool.get();
    assert_eq!(clone.idle_len(), 0);
    drop(buf);
    assert_eq!(clone.idle_len(), 1, "state is shared, not copied");

    // stats are shared too, and the borrowed-view kind of clone (via &*clone)
    // sees the same counters
    assert_eq!(clone.stats().leases, 1);
}

#[test]
fn sequential_leases_recycle_one_buffer() {
    let pool = BufferPool::new(|| vec![0u32; 64]);
    for k in 0..100 {
        pool.get()[0] = k; // lease, touch, drop — all in one statement
    }
    let stats = pool.stats();
    assert_eq!(stats.leases, 100);
    assert_eq!(stats.allocations, 1);
    assert_eq!(stats.reuses(), 99);
}

#[test]
fn rayon_stress_many_tasks() {
    let ntasks = 20_000;
    let pool = BufferPool::new(|| vec![0u64; 64]);
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
    assert_eq!(stats.leases, ntasks);
    assert!(stats.allocations >= 1, "work actually happened");
    // Deliberately no upper bound on `allocations`: rayon assigns jobs, not
    // resources, to threads (same guarantee class as `map_init`'s per-job
    // init), so the count depends on scheduling — only `leases` is
    // deterministic here. What *is* deterministic: every allocated buffer
    // came back to the shared pile.
    assert_eq!(
        pool.idle_len(),
        stats.allocations,
        "every allocated buffer was returned, none lost"
    );
}

#[test]
fn non_static_initializer_with_scoped_threads() {
    // The initializer borrows `dims` (a stack local, not 'static, not even
    // Copy-able in spirit — a real caller might pass a config struct here).
    struct Dims {
        rows: usize,
        cols: usize,
    }
    let dims = Dims { rows: 32, cols: 48 };
    let pool = BufferPool::new(|| vec![0.0f64; dims.rows * dims.cols]);

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
    // `dims` still alive and borrowed here; pool dropped after it
    assert_eq!(dims.rows, 32);
}

#[test]
fn guard_moves_across_threads_mid_lease() {
    // A lease taken on the main thread can finish on a worker thread.
    let pool = BufferPool::new(|| vec![0u32; 8]);
    let mut buf = pool.get();
    buf[0] = 7;
    let pool_handle = pool.clone();

    thread::scope(|s| {
        s.spawn(move || {
            assert_eq!(buf[0], 7);
            drop(buf); // returned from this thread
        });
    });
    assert_eq!(pool_handle.idle_len(), 1);
}

#[test]
fn debug_impls_do_not_panic() {
    let pool = BufferPool::new(|| vec![1.0f64; 2]).with_max_idle(4);
    let guard = pool.get();
    let _ = format!("{pool:?} {guard:?} {:?}", pool.stats());
}

#[test]
fn pooled_works_for_non_default_and_odd_types() {
    // T needs no Default/Clone/Debug — only the initializer's output.
    struct Opaque(#[allow(dead_code)] u64);
    let pool = BufferPool::new(|| Opaque(3));
    let _ = pool.get();
    let _ = pool.get();
    assert_eq!(pool.stats().allocations, 1);
}
