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

use anyhow::Result;
use mpi::collective::CommunicatorCollectives;
use rand::{RngExt, SeedableRng};
use rand::rngs::ChaCha8Rng;
use std::iter::zip;

use sope::{
    comm::WorldComm,
    cond_info, cond_println, ensure_eq,
    reduction::{all_of, any_of},
    sort::{is_sorted, is_sorted_by, sort, sort_by},
    traits::GEquivalence,
};

#[derive(GEquivalence, Debug, Default, Clone, Eq, PartialEq, PartialOrd, Ord)]
struct Triple<T1> {
    first: T1,
    second: T1,
    third: T1,
}

#[derive(GEquivalence, Debug, Clone, Default)]
struct GenericTriple<T1, T2, T3> {
    first: T1,
    second: T2,
    third: T3,
}

fn test_triple(c: &WorldComm) -> Result<()> {
    let nelts = 125;
    let rng = ChaCha8Rng::seed_from_u64(0);
    let rng_nums: Vec<i32> = rng.random_iter::<i32>().take(nelts * 3).collect();
    let mut lvec: Vec<Triple<i32>> = rng_nums
        .chunks_exact(3)
        .flat_map(|cx| {
            if let [x, y, z] = cx {
                Some(Triple::<i32> {
                    first: *x,
                    second: *y,
                    third: *z,
                })
            } else {
                None
            }
        })
        .collect();

    sort(&mut lvec, &c.comm)?;
    let local_sorted = lvec.is_sorted();
    ensure_eq!(all_of(local_sorted, &c.comm), true);

    let d_sorted = is_sorted(&lvec, &c.comm)?;
    ensure_eq!(d_sorted, true);
    Ok(())
}

fn test_generic_triple(c: &WorldComm) -> Result<()> {
    let nelts = 125;
    type TTuple = GenericTriple<i32, u16, f32>;
    let irng = ChaCha8Rng::seed_from_u64(0).random_iter::<i32>();
    let urng = ChaCha8Rng::seed_from_u64(0).random_iter::<u16>();
    let frng = ChaCha8Rng::seed_from_u64(0).random_iter::<f32>();

    let mut lvec: Vec<TTuple> = zip(irng, urng)
        .zip(frng)
        .take(nelts)
        .map(|((x, y), z)| TTuple {
            first: x,
            second: y,
            third: z,
        })
        .collect();

    let cmp_fn =
        |a: &TTuple, b: &TTuple| (a.first, a.second).cmp(&(b.first, b.second));

    sort_by(&mut lvec, cmp_fn, &c.comm)?;
    let local_sorted = lvec.is_sorted_by_key(|a| (a.first, a.second));
    ensure_eq!(local_sorted, true);

    let d_sorted = is_sorted_by(&lvec,|a,b| cmp_fn(a,b).is_le(), &c.comm)?;
    ensure_eq!(d_sorted, true);

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
        cond_info!(c.is_root(); "{} SUCCESSFUL", tm );
    }
}

fn run(c: &WorldComm) {
    let _ = env_logger::try_init();
    log_if_error(test_triple(c), c, "DERIVED STRUCT SORT");
    log_if_error(test_generic_triple(c), c, "GENERIC DERIVED STRUCT SORT");
    c.comm.barrier();
    cond_println!(c.is_root(); "DERIVE TEST COMPLETED");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
