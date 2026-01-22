use anyhow::{Ok, Result};
use mpi::traits::Equivalence;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{cmp::Ordering, iter::zip};

use sope::{
    collective::{gatherv_full_vec, gatherv_vec},
    comm::WorldComm,
    cond_println, ensure_eq,
    reduction::{all_of, any_of},
    shift::right_shift,
    sort::{
        bitonic_sort, is_sorted, is_sorted_by, sort, sort_by, stable_sort_by,
    },
};

fn test_sample_sort(c: &WorldComm) -> Result<()> {
    let rng = ChaCha8Rng::seed_from_u64(0);
    let mut v: Vec<i32> = rng.random_iter::<i32>().take(230).collect();

    //before sorting
    let bsrt = is_sorted(&v, &c.comm)?;
    ensure_eq!(bsrt, false);

    //after sorting
    sort(&mut v, &c.comm)?;
    let bsrt = is_sorted(&v, &c.comm)?;
    ensure_eq!(bsrt, true);
    Ok(())
}

fn test_bitonic_sort(c: &WorldComm) -> Result<()> {
    let n: usize = 10;
    let rng = ChaCha8Rng::seed_from_u64(133 * c.rank as u64);
    let mut v: Vec<i32> = rng
        .random_iter::<i32>()
        .take(n)
        .map(|x| (x % 10000).abs())
        .collect();

    // have ground truth
    let truthv = gatherv_full_vec(&v, 0, &c.comm)?.map(|mut x| {
        x.sort();
        x
    });

    bitonic_sort(&mut v, i32::cmp, &c.comm)?;

    let bsrt = v.as_slice().is_sorted();
    ensure_eq!(bsrt, true);

    let last = v.last().unwrap_or(&0);
    let prev = right_shift(last, &c.comm);
    let ind = all_of(c.is_root() || prev <= v[0], &c.comm);
    if c.rank > 0 {
        sope::ensure!(prev <= v[0]);
    }
    if !ind {
        return Ok(());
    }

    let bsrt = is_sorted(&v, &c.comm)?;
    ensure_eq!(bsrt, true);

    let allsortedv = gatherv_full_vec(&v, 0, &c.comm)?;
    if let (Some(truthv), Some(allsortedv)) = (truthv, allsortedv) {
        for (a, b) in zip(truthv, allsortedv) {
            ensure_eq!(a, b);
        }
    }

    Ok(())
}

#[derive(Equivalence, Clone, Default, Debug, Eq, PartialEq)]
struct IPair {
    first: i32,
    second: i32,
}

fn test_sort_imbalanced(c: &WorldComm) -> Result<()> {
    let rng = ChaCha8Rng::seed_from_u64(13 * c.rank as u64);
    let n = 10 + c.rank as usize * 3;
    let rvec: Vec<i32> = rng.random_iter::<i32>().take(2 * n).collect();
    let mut v: Vec<IPair> = rvec
        .chunks(2)
        .map(|sa| IPair {
            first: sa[0],
            second: sa[1],
        })
        .collect();
    let ip_cmp = |x: &IPair, y: &IPair| {
        if x == y {
            Ordering::Equal
        } else if x.first < y.first || (x.first == y.first && x.second < y.second)
        {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    };

    // For testing
    let truthv = gatherv_full_vec(&v, 0, &c.comm)?.map(|mut x| {
        x.sort_by(ip_cmp);
        x
    });
    let rcv_sizes: Option<Vec<i32>> = if c.rank == 0 {
        Some((0..c.size).map(|j| 10 + j * 3).collect())
    } else {
        None
    };

    //sort
    sort_by(&mut v, ip_cmp, &c.comm)?;

    let allsortedv =
        gatherv_vec(&v[..], rcv_sizes.as_ref().map(|x| &x[..]), 0, &c.comm)?;
    let bsrt = v
        .as_slice()
        .is_sorted_by(|x, y| ip_cmp(x, y) == Ordering::Less);
    ensure_eq!(bsrt, true);

    let bsrt = is_sorted_by(&v, |x, y| ip_cmp(x, y).is_le(), &c.comm)?;
    ensure_eq!(bsrt, true);

    if c.is_root() {
        sope::ensure!(truthv.is_some());
        sope::ensure!(allsortedv.is_some());
        if let (Some(truthv), Some(allsortedv)) = (truthv, allsortedv) {
            ensure_eq!(truthv.len(), allsortedv.len());
            for (x, y) in zip(truthv, allsortedv) {
                ensure_eq!(x, y);
            }
        }
    }

    Ok(())
}

fn test_stable_sort(c: &WorldComm) -> Result<()> {
    let n = 100 + c.rank;
    let prefix: i32 = if c.rank == 0 {
        0
    } else {
        (100 * (c.rank) + ((c.rank + 1) * c.rank) / 2) as usize
    } as i32;
    let rng = ChaCha8Rng::seed_from_u64(7 * c.rank as u64);
    let mut v: Vec<IPair> = (0..n)
        .zip(rng.random_iter::<i32>().take(n as usize))
        .map(|(a, b)| IPair {
            first: a + prefix,
            second: b % 4,
        })
        .collect();
    let ip_cmp = |x: &IPair, y: &IPair| x.second.cmp(&y.second);

    stable_sort_by(&mut v, ip_cmp, &c.comm)?;
    // assert it is sorted lexicographically by both (second, first)
    let full_cmp = |x: &IPair, y: &IPair| {
        x.second < y.second || (x.second == y.second && x.first < y.first)
    };

    let bsorted = is_sorted_by(&v, full_cmp, &c.comm)?;
    ensure_eq!(bsorted, true);

    Ok(())
}

fn log_if_error<T>(ex: Result<T>, c: &WorldComm, tm: &str) {
    if any_of(ex.is_err(), &c.comm) {
        //println!("{}", ex.map_or_else(|e| e.to_string(), |_r| "".to_string()))
        sope::gather_error!(
            &c.comm; "{}",
            ex.map_or_else(|e| e.to_string(), |_r| "".to_string())
        );
    } else {
        cond_println!(c.is_root(); "{} SUCCESSFUL", tm );
    }
}

fn run(c: &WorldComm) {
    let _ = env_logger::try_init();
    log_if_error(test_stable_sort(c), c, "STABLE SORT");
    log_if_error(test_sort_imbalanced(c), c, "SAMPLE SORT IMBALANCED");
    log_if_error(test_sample_sort(c), c, "SAMPLE SORT");
    log_if_error(test_bitonic_sort(c), c, "BITONIC SORT");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
