//
// Copyright 2026 Georgia Institute of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

use anyhow::{Ok, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sope::{
    bcast::{bcast_one, bcast_vec},
    comm::WorldComm,
    cond_println, ensure_eq,
    reduction::any_of,
};

fn test_bcast_one(c: &WorldComm) -> Result<()> {
    let s_in = if c.rank == 0 { Some(240i32) } else { None };
    let s_out = bcast_one(s_in, 0, &c.comm)?;
    ensure_eq!(s_out, 240i32);
    Ok(())
}

fn test_bcast_vec(c: &WorldComm) -> Result<()> {
    let nsend = 25;
    let rng = ChaCha8Rng::seed_from_u64(0);
    let s_vec: Vec<u32> = rng.random_iter::<u32>().take(nsend).collect();
    let s_in = if c.rank == c.size - 1 {
        Some(&s_vec[..])
    } else {
        None
    };
    let s_out = bcast_vec(s_in, c.size - 1, &c.comm)?;

    ensure_eq!(s_vec.len(), s_out.len());
    for (a, b) in s_vec.iter().zip(s_out.iter()) {
        ensure_eq!(*a, *b);
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
    log_if_error(test_bcast_one(c), c, "BROADCAST ONE");
    log_if_error(test_bcast_vec(c), c, "BROADCAST VEC");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
