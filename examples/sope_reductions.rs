use anyhow::{Ok, Result};
use mpi::collective::{SystemOperation, UserOperation};
use sope::{
    comm::WorldComm,
    cond_info, ensure_eq,
    reduction::{
        all_of, allreduce_vec, any_of, exclusive_scan, max_element, min_element, none_of, reduce,
        reduce_vec, scan,
    },
};

fn test_reduce_one(c: &WorldComm) -> Result<()> {
    // min
    let x: i32 = -13 * (c.size - c.rank);
    let y = reduce(&x, c.size / 2, &c.comm, SystemOperation::min());
    if c.rank == c.size / 2 {
        ensure_eq!(y, Some(-13 * c.size));
    } else {
        ensure_eq!(y, None);
    }

    // Sum
    let y = reduce(&3, 0, &c.comm, SystemOperation::sum());
    if c.rank == 0 {
        ensure_eq!(y, Some(3 * c.size));
    } else {
        ensure_eq!(y, None);
    }

    // Custom Sum
    let y = reduce(
        &2i32,
        c.size - 1,
        &c.comm,
        &UserOperation::commutative(|a, b| {
            let x: &[i32] = a.downcast().unwrap();
            let y: &mut [i32] = b.downcast().unwrap();
            y[0] += x[0]
        }),
    );
    if c.rank == c.size - 1 {
        ensure_eq!(y, Some(2 * c.size));
    } else {
        ensure_eq!(y, None);
    }
    Ok(())
}

fn test_reduce_vec(c: &WorldComm) -> Result<()> {
    let n = 13i32;
    let v: Vec<i32> = (0..n).map(|x| x + c.rank).collect();
    let ranksum: i32 = (c.size * (c.size - 1)) / 2;

    let rvec = reduce_vec(&v, c.size / 2, &c.comm, SystemOperation::sum());
    if c.rank == c.size / 2 {
        assert!(rvec.is_some());
        if let Some(rvec) = rvec {
            ensure_eq!(rvec.len(), n as usize);
            let tvec: Vec<i32> = (0..n).map(|i| ranksum + i * c.size).collect();
            ensure_eq!(rvec, tvec);
        }
    } else {
        ensure_eq!(rvec, None);
    }

    Ok(())
}

fn test_all_reduce(c: &WorldComm) -> Result<()> {
    let n = 10i32; // numbers per rank
    let v: Vec<i32> = (0..n).collect();
    let w: Vec<i32> = allreduce_vec(&v, &c.comm, SystemOperation::sum());
    ensure_eq!(n as usize, w.len());
    for (i, rw) in w.iter().enumerate() {
        ensure_eq!(c.size * i as i32, *rw);
    }

    Ok(())
}

fn test_scan(c: &WorldComm) -> Result<()> {
    let r = c.rank * 2;
    let g = scan(&r, &c.comm, SystemOperation::sum());
    ensure_eq!(c.rank * (c.rank + 1), g);

    let m = exclusive_scan(&r, &c.comm, SystemOperation::max());
    if r != 0 {
        ensure_eq!((c.rank - 1) * 2, m);
    }

    Ok(())
}

fn test_arg(c: &WorldComm) -> Result<()> {
    let maxloc = max_element(&(3 * c.rank), &c.comm);
    ensure_eq!(c.size - 1, maxloc.0);
    ensure_eq!(3 * (c.size - 1), maxloc.1);

    let minloc = min_element(&(13 + c.rank), &c.comm);
    ensure_eq!(0, minloc.0);
    ensure_eq!(13, minloc.1);
    Ok(())
}

fn test_logical(c: &WorldComm) -> Result<()> {
    let rt = all_of(true, &c.comm);
    sope::ensure!(rt);
    let rt = any_of(c.rank == 0, &c.comm);
    sope::ensure!(rt);
    let rt = none_of(c.rank == 0, &c.comm);
    sope::ensure!(rt);
    Ok(())
}

fn log_if_error<T>(ex: Result<T>, c: &WorldComm, tm: &str) {
    if any_of(ex.is_err(), &c.comm) {
        sope::gather_error!(
            &c.comm; "{}",
            ex.map_or_else(|e| e.to_string(), |_r| "".to_string())
        );
    } else {
        cond_info!(c.is_root(); "{} SUCCESSFUL", tm );
    }
}

fn run(c: &WorldComm) {
    let _ = env_logger::try_init();

    //
    log_if_error(test_reduce_one(c), c, "REDUCE ONE");
    log_if_error(test_reduce_vec(c), c, "REDUCE VEC");
    log_if_error(test_all_reduce(c), c, "ALLREDUCE");
    log_if_error(test_scan(c), c, "SCAN");
    log_if_error(test_arg(c), c, "ELEMENT");
    log_if_error(test_logical(c), c, "LOGICAL" );
    cond_info!(c.is_root(); "TEST COMPLETED");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
