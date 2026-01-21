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

use std::fmt::Debug;

use anyhow::Result;
use mpi::{collective::SystemOperation, traits::Equivalence};
use sope::{
    comm::WorldComm,
    cond_println,
    distribution::{
        Distributor, StableDistributor, distribute_vec, stable_distribute_vec,
    },
    ensure_eq, gather_info,
    partition::{Dist, ModuloDist},
    reduction::{allreduce_sum, any_of, exclusive_scan},
};

#[derive(Debug, Equivalence, Default, Clone, Copy, PartialEq, Eq)]
struct A2Pair {
    a: i32,
    b: i32,
}

impl A2Pair {
    pub fn from_usize(x: usize) -> Self {
        Self {
            a: x as i32,
            b: x as i32,
        }
    }
}

#[allow(dead_code)]
fn test_distributor<T>(
    tv: &[T],
    mpart: &dyn Dist,
    c: &WorldComm,
) -> Result<Vec<T>>
where
    T: 'static + Equivalence + Default + Clone + Debug + Eq + Sync + Send,
{
    let mut eqv: Vec<T> = vec![T::default(); mpart.local_size()];
    let sdist = StableDistributor::new(mpart, &c.comm);
    let cca = sdist.cc_args(tv);
    gather_info!(&c.comm; "{:?}", cca );
    sdist.distribute(tv, &mut eqv)?;
    Ok(eqv)
}

fn test_distribute_helper<T, F>(
    dsize: usize,
    gen_fn: F,
    c: &WorldComm,
) -> Result<()>
where
    T: 'static + Equivalence + Default + Clone + Debug + Eq + Sync + Send,
    F: Fn(usize) -> T,
{
    let prefix: usize = exclusive_scan(&dsize, &c.comm, SystemOperation::sum());
    let total_size = allreduce_sum(&dsize, &c.comm);
    let v: Vec<T> = (0..dsize).map(|x| gen_fn(prefix + x)).collect();

    let mpart = ModuloDist::new(total_size, c.size, c.rank);
    let eqv = distribute_vec(&v, &mpart, &c.comm)?;

    // get expected size
    let eq_size: usize = total_size / c.size as usize
        + ((c.rank as usize) < (total_size % c.size as usize)) as usize;
    let eq_prefix: usize =
        exclusive_scan(&eq_size, &c.comm, SystemOperation::sum());

    ensure_eq!(eq_size, eqv.len());
    for (i, vx) in eqv.iter().enumerate() {
        let expected: T = gen_fn(eq_prefix + i);
        ensure_eq!(expected, vx.clone());
    }
    Ok(())
}

fn test_stable_distribute_helper<T, F>(
    dsize: usize,
    gen_fn: F,
    c: &WorldComm,
) -> Result<()>
where
    T: 'static + Equivalence + Default + Clone + Debug + Eq + Sync + Send + Copy,
    F: Fn(usize) -> T,
{
    let prefix: usize = exclusive_scan(&dsize, &c.comm, SystemOperation::sum());
    let total_size = allreduce_sum(&dsize, &c.comm);

    let v: Vec<T> = (0..dsize).map(|x| gen_fn(prefix + x)).collect();

    let mpart = ModuloDist::new(total_size, c.size, c.rank);
    let eqv = stable_distribute_vec(&v, &mpart, &c.comm)?;

    let eq_size: usize = total_size / c.size as usize
        + ((c.rank as usize) < (total_size % c.size as usize)) as usize;
    let eq_prefix: usize =
        exclusive_scan(&eq_size, &c.comm, SystemOperation::sum());

    ensure_eq!(eq_size, eqv.len());
    for (i, vx) in eqv.iter().enumerate() {
        let expected: T = gen_fn(eq_prefix + i);
        ensure_eq!(expected, *vx);
    }

    Ok(())
}

fn test_distribute(c: &WorldComm) -> Result<()> {
    let gen_fn = A2Pair::from_usize;
    // create unequal distribution
    let size: usize = usize::max(10, 100 - 10 * c.rank as usize);
    test_distribute_helper(size, gen_fn, c)?;
    test_stable_distribute_helper(size, gen_fn, c)?;

    let size = if c.rank == c.size / 2 {
        10 * c.size as usize
    } else {
        0usize
    };
    test_distribute_helper(size, gen_fn, c)?;
    test_stable_distribute_helper(size, gen_fn, c)?;

    let gen_fn = |x: usize| x as i32;
    // create unequal distribution
    let size: usize = usize::max(10, 100 - 10 * c.rank as usize);
    test_distribute_helper(size, gen_fn, c)?;
    test_stable_distribute_helper(size, gen_fn, c)?;

    let size = if c.rank == c.size / 2 {
        10 * c.size as usize
    } else {
        0usize
    };
    test_distribute_helper(size, gen_fn, c)?;
    test_stable_distribute_helper(size, gen_fn, c)?;

    Ok(())
}

fn log_if_error<T>(ex: Result<T>, c: &WorldComm, tm: &str) {
    if any_of(ex.is_err(), &c.comm) {
        sope::gather_eprintln!(
            &c.comm; "{}",
            ex.map_or_else(|e| e.to_string(), |_r| "".to_string())
        );
    } else {
        cond_println!(c.is_root(); "{} TEST SUCCESSFUL", tm );
    }
}

fn run(c: &WorldComm) {
    let _ = env_logger::try_init();
    log_if_error(test_distribute(c), c, "DISTRIBUTE");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
