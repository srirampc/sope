use anyhow::Result;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sope::{comm::WorldComm, cond_println, ensure_eq, reduction::any_of, shift::{left_shift, right_shift}};

fn test_shift(c: &WorldComm) -> Result<()> {
    let rng = ChaCha8Rng::seed_from_u64(0);
    let rints: Vec<i32> = rng.random_iter::<i32>().take(c.size as usize).collect();
    let x = rints[c.rank as usize];

    let lx = left_shift(&x, &c.comm);
    if c.rank < c.size - 1 {
        ensure_eq!(lx, rints[c.rank as usize + 1]);
    }

    let rx = right_shift(&x, &c.comm);
    if c.rank > 0 {
        ensure_eq!(rx, rints[c.rank as usize - 1]);
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
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
