use anyhow::Result;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sope::{
    comm::WorldComm,
    cond_println, ensure_eq,
    reduction::any_of,
    shift::{left_shift, left_shift_vec, right_shift, right_shift_vec},
};

fn test_shift(c: &WorldComm) -> Result<()> {
    let rng = ChaCha8Rng::seed_from_u64(0);
    let rints: Vec<i32> =
        rng.random_iter::<i32>().take(c.size as usize).collect();
    let x = rints[c.rank as usize];

    let lx = left_shift(&x, &c.comm);
    if c.rank < c.size - 1 {
        ensure_eq!(lx.unwrap(), rints[c.rank as usize + 1]);
    }

    let rx = right_shift(&x, &c.comm);
    if c.rank > 0 {
        ensure_eq!(rx.unwrap(), rints[c.rank as usize - 1]);
    }
    Ok(())
}

fn test_shift_vec(c: &WorldComm) -> Result<()> {
    let snd_size = 10 + c.rank as usize * 2;
    let rng = ChaCha8Rng::seed_from_u64(c.rank as u64);
    let send_ints: Vec<i32> = rng.random_iter::<i32>().take(snd_size).collect();
    let lx = left_shift_vec(&send_ints, &c.comm);
    if c.rank < c.size - 1 {
        let rcv_size = 10 + (c.rank + 1) as usize * 2;
        let lx = lx.unwrap_or_default();
        ensure_eq!(lx.len(), rcv_size);
        let rng2 = ChaCha8Rng::seed_from_u64((c.rank + 1) as u64);
        let rcv_ints: Vec<i32> =
            rng2.random_iter::<i32>().take(rcv_size).collect();
        for (a, b) in lx.iter().zip(rcv_ints.iter()) {
            ensure_eq!(*a, *b);
        }
    }

    let rx = right_shift_vec(&send_ints, &c.comm);
    if c.rank > 0 {
        let rcv_proc = c.rank as usize - 1;
        let rcv_size = 10 + rcv_proc * 2;
        let rx = rx.unwrap_or_default();
        ensure_eq!(rx.len(), rcv_size);
        let rng2 = ChaCha8Rng::seed_from_u64(rcv_proc as u64);
        let rcv_ints: Vec<i32> =
            rng2.random_iter::<i32>().take(rcv_size).collect();
        for (a, b) in rx.iter().zip(rcv_ints.iter()) {
            ensure_eq!(*a, *b);
        }
    }
    Ok(())
}

fn log_if_error<T>(ex: Result<T>, c: &WorldComm, tm: &str) {
    if any_of(ex.is_err(), &c.comm) {
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
    log_if_error(test_shift(c), c, "SHIFT");
    log_if_error(test_shift_vec(c), c, "SHIFT VEC");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
