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

use std::iter::{repeat_n, zip};

use anyhow::{Ok, Result};
use mpi::traits::{Communicator, Equivalence};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sope::{
    collective::{
        All2allvArgs, all2all, all2all_vec, all2allv, all2allv_vec, allgather,
        allgather_one, allgather_vec, allgatherv, allgatherv_vec, gather,
        gather_one, gather_vec, gatherv, gatherv_vec, scatter, scatter_one,
        scatter_vec, scatterv, scatterv_vec,
    },
    comm::WorldComm,
    cond_println, ensure_eq,
    reduction::any_of,
};

#[derive(Debug, Equivalence, Default, Clone)]
struct CPair {
    a: i32,
    b: f32,
}

impl CPair {
    pub fn new(a: i32, b: f32) -> Self {
        Self { a, b }
    }
}

#[derive(Debug, Equivalence, Default, Clone)]
struct A2Pair {
    a: i32,
    b: i32,
}

impl A2Pair {
    pub fn new(a: i32, b: i32) -> Self {
        Self { a, b }
    }
}

fn test_scatter_one(c: &WorldComm) -> Result<()> {
    let v: Option<Vec<i32>> = if c.rank == 0 {
        Some((0..c.size).map(|i| 3 * i * i).collect())
    } else {
        None
    };
    let my = scatter_one(v.as_ref().map(|x| &x[..]), 0, &c.comm)?;
    ensure_eq!(3 * c.rank * c.rank, my);
    let v: Option<Vec<CPair>> = if c.rank == 0 {
        Some(
            (0..c.size)
                .map(|i| CPair::new(-2 * i, 6.22 * (i * i) as f32))
                .collect(),
        )
    } else {
        None
    };
    let result = scatter_one(v.as_ref().map(|x| &x[..]), 0, &c.comm)?;
    ensure_eq!(-2 * c.rank, result.a);
    sope::ensure!((result.b - 6.22 * (c.rank * c.rank) as f32).abs() < 1e-12);
    Ok(())
}

fn test_scatter(c: &WorldComm) -> Result<()> {
    let msgsize: isize = 13;
    let v: Option<Vec<isize>> = if c.rank == 0 {
        Some(
            (0..c.size as isize)
                .flat_map(|i| (0..msgsize).map(move |j| -2 * j + i))
                .collect(),
        )
        //(0..c.size).map(|i| 3 * i * i).collect()
    } else {
        None
    };

    let mut v1 = vec![-1isize; msgsize as usize];
    scatter(v.as_ref().map(|x| &x[..]), &mut v1, 0, &c.comm)?;
    for j in 0..msgsize {
        ensure_eq!(v1[j as usize], -2 * j + c.rank as isize);
    }

    let v2 = scatter_vec(v.as_ref().map(|x| &x[..]), 0, &c.comm)?;
    ensure_eq!(v2.len(), msgsize as usize);
    for j in 0..msgsize {
        ensure_eq!(v2[j as usize], -2 * j + c.rank as isize);
    }
    Ok(())
}

fn test_scatterv(c: &WorldComm) -> Result<()> {
    let (v, s): (Option<Vec<usize>>, Option<Vec<i32>>) = if c.rank == 0 {
        (
            Some(
                (0..c.size as usize)
                    .flat_map(|i| (0..i + 1).map(move |j| 2 * j + 3 * i))
                    .collect(),
            ),
            Some((1..c.size + 1).collect()),
        )
    } else {
        (None, None)
    };

    let mut v1 = vec![0usize; (1 + c.rank) as usize];
    scatterv(
        v.as_ref().map(|x| &x[..]),
        &mut v1,
        s.as_ref().map(|x| &x[..]),
        0,
        &c.comm,
    )?;
    for (j, vj) in v1.iter().enumerate().take(1 + c.rank as usize) {
        ensure_eq!(*vj, 2 * j + 3 * c.rank as usize);
    }

    let v2 = scatterv_vec(
        v.as_ref().map(|x| &x[..]),
        s.as_ref().map(|x| &x[..]),
        0,
        &c.comm,
    )?;
    ensure_eq!(v2.len(), 1 + c.rank as usize);
    for (j, vj) in v2.iter().enumerate().take(1 + c.rank as usize) {
        ensure_eq!(*vj, 2 * j + 3 * c.rank as usize);
    }
    Ok(())
}

fn test_gather_one(c: &WorldComm) -> Result<()> {
    let x = CPair::new(13 * c.rank, std::f32::consts::PI / c.rank as f32);
    let gx = gather_one(&x, 0, &c.comm)?;
    if c.rank == 0 {
        ensure_eq!(gx.is_some(), true);
        if let Some(gv) = gx {
            ensure_eq!(gv.len(), c.comm.size() as usize);
            for (j, vj) in gv.iter().enumerate().take(c.size as usize) {
                ensure_eq!(vj.a, 13 * j as i32);
                ensure_eq!(vj.b, std::f32::consts::PI / j as f32);
            }
        }
    } else {
        sope::ensure!(gx.is_none());
    }

    Ok(())
}

fn test_gather(c: &WorldComm) -> Result<()> {
    let msize: usize = 13;
    let els: Vec<u32> =
        (0..msize).map(|j| j as u32 * 3 * c.rank as u32).collect();

    let mut all = if c.rank == c.size - 1 {
        Some(vec![0u32; msize * c.size as usize])
    } else {
        None
    };

    gather(&els, all.as_mut().map(|x| &mut x[..]), c.size - 1, &c.comm)?;
    if c.rank == c.size - 1 {
        sope::ensure!(all.is_some());
        if let Some(gv) = all.as_ref() {
            for i in 0..c.size as usize {
                for (j, vj) in
                    gv[i * msize..(i * msize + msize)].iter().enumerate()
                {
                    assert_eq!(*vj, j as u32 * 3 * i as u32);
                }
            }
        }
    } else {
        sope::ensure!(all.is_none());
    }

    let all = gather_vec(&els, c.size - 1, &c.comm)?;
    if c.rank == c.size - 1 {
        sope::ensure!(all.is_some());
        if let Some(gv) = all.as_ref() {
            assert_eq!(gv.len(), msize * c.size as usize);
            for i in 0..c.size as usize {
                for (j, vj) in
                    gv[i * msize..(i * msize + msize)].iter().enumerate()
                {
                    assert_eq!(*vj, j as u32 * 3 * i as u32);
                }
            }
        }
    } else {
        sope::ensure!(all.is_none());
    }
    Ok(())
}

fn test_gatherv(c: &WorldComm) -> Result<()> {
    let size: usize = 5 * (c.rank as usize + 2);
    let els: Vec<i32> = (0..size as i32).map(|i| c.rank * 13 - 42 * i).collect();

    let (recv_sizes, mut all, total_size) = if c.rank == 0 {
        let sz: Vec<i32> = (0..c.size).map(|i| 5 * (i + 2)).collect();
        let nsz = sz.iter().sum::<i32>() as usize;
        (Some(sz), Some(vec![0i32; nsz]), nsz)
    } else {
        (None, None, 0)
    };
    gatherv(
        &els,
        all.as_mut().map(|x| &mut x[..]),
        recv_sizes.as_ref().map(|x| &x[..]),
        0,
        &c.comm,
    )?;
    if c.rank == 0 {
        sope::ensure!(all.is_some());
        if let Some(gv) = all.as_ref() {
            let mut pos: usize = 0;
            for i in 0..c.size {
                for j in 0..(5 * (i + 2)) {
                    ensure_eq!(gv[pos], i * 13 - 42 * j);
                    pos += 1;
                }
            }
        }
    } else {
        sope::ensure!(all.is_none())
    }
    let all = gatherv_vec(&els, recv_sizes.as_ref().map(|x| &x[..]), 0, &c.comm)?;
    if c.rank == 0 {
        sope::ensure!(all.is_some());
        if let Some(gv) = all.as_ref() {
            sope::ensure!(!gv.is_empty());
            ensure_eq!(gv.len(), total_size);
            let mut pos: usize = 0;
            for i in 0..c.size {
                for j in 0..(5 * (i + 2)) {
                    ensure_eq!(gv[pos], i * 13 - 42 * j);
                    pos += 1;
                }
            }
        }
    } else {
        sope::ensure!(all.is_none())
    }

    Ok(())
}

fn test_allgather_one(c: &WorldComm) -> Result<()> {
    let x = CPair::new(13 * c.rank, std::f32::consts::PI / c.rank as f32);
    let gv = allgather_one(&x, &c.comm)?;
    ensure_eq!(gv.len(), c.comm.size() as usize);
    for (j, vj) in gv.iter().enumerate().take(c.size as usize) {
        ensure_eq!(vj.a, 13 * j as i32);
        ensure_eq!(vj.b, std::f32::consts::PI / j as f32);
    }

    Ok(())
}

fn test_allgather(c: &WorldComm) -> Result<()> {
    let msize: usize = 13;
    let els: Vec<u32> =
        (0..msize).map(|j| j as u32 * 3 * c.rank as u32).collect();

    let mut all = vec![0u32; msize * c.size as usize];
    allgather(&els, all.as_mut(), &c.comm)?;
    for i in 0..c.size as usize {
        for (j, vj) in all[i * msize..(i * msize + msize)].iter().enumerate() {
            assert_eq!(*vj, j as u32 * 3 * i as u32);
        }
    }

    let all = allgather_vec(&els, &c.comm)?;
    assert_eq!(all.len(), msize * c.size as usize);
    for i in 0..c.size as usize {
        for (j, vj) in all[i * msize..(i * msize + msize)].iter().enumerate() {
            assert_eq!(*vj, j as u32 * 3 * i as u32);
        }
    }

    Ok(())
}

fn test_allgatherv(c: &WorldComm) -> Result<()> {
    let size: usize = 5 * (c.rank as usize + 2);
    let els: Vec<i32> = (0..size as i32).map(|i| c.rank * 13 - 42 * i).collect();

    let recv_sizes: Vec<i32> = (0..c.size).map(|i| 5 * (i + 2)).collect();
    let total_size: usize = recv_sizes.iter().sum::<i32>() as usize;
    let mut all: Vec<i32> = vec![0i32; total_size];

    allgatherv(&els, &mut all, &recv_sizes, &c.comm)?;
    let mut pos: usize = 0;
    for i in 0..c.size {
        for j in 0..(5 * (i + 2)) {
            ensure_eq!(all[pos], i * 13 - 42 * j);
            pos += 1;
        }
    }

    //
    let all = allgatherv_vec(&els, recv_sizes.as_ref(), &c.comm)?;
    sope::ensure!(!all.is_empty());
    ensure_eq!(all.len(), total_size);
    let mut pos: usize = 0;
    for i in 0..c.size {
        for j in 0..(5 * (i + 2)) {
            ensure_eq!(all[pos], i * 13 - 42 * j);
            pos += 1;
        }
    }

    Ok(())
}

fn test_all2all(c: &WorldComm) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let msgs: Vec<A2Pair> = (0..c.size)
        .map(|i| A2Pair::new(c.rank * i, rng.random::<i32>()))
        .collect();
    // test one
    let mut result = vec![A2Pair::new(0, 0); c.size as usize];
    all2all(&msgs, &mut result, &c.comm)?;

    for (i, rj) in result.iter_mut().enumerate() {
        ensure_eq!(i as i32 * c.rank, rj.a);
        rj.b /= 13;
    }

    let result2 = all2all_vec(&result, &c.comm)?;
    ensure_eq!(result2.len(), c.size as usize);
    for (i, (m, r)) in zip(msgs.iter(), result2.iter()).enumerate() {
        ensure_eq!(m.a, i as i32 * c.rank);
        ensure_eq!(m.b / 13, r.b);
    }

    let msize: i32 = 230;
    let msgs: Vec<i32> = (0..c.size)
        .flat_map(|i| (0..msize).map(move |j| i * c.rank * 33 - 2 * j))
        .collect();
    let result = all2all_vec(&msgs, &c.comm)?;
    ensure_eq!(msize * c.size, result.len() as i32);
    for i in 0..c.size {
        for j in 0..msize {
            ensure_eq!(i * c.rank * 33 - 2 * j, result[(i * msize + j) as usize])
        }
    }

    Ok(())
}

fn test_all2allv(c: &WorldComm) -> Result<()> {
    let rng = ChaCha8Rng::seed_from_u64(0);
    let send_counts: Vec<i32> = (0..c.size)
        .map(|i| 2 * c.rank + 3 * (c.size - i - 1))
        .collect();
    let recv_counts: Vec<i32> = (0..c.size)
        .map(|i| 2 * i + 3 * (c.size - c.rank - 1))
        .collect();
    let nsend = send_counts.iter().sum::<i32>() as usize;
    let nrecv = recv_counts.iter().sum::<i32>() as usize;
    let a_itr = (0..c.size)
        .flat_map(|i| repeat_n(c.rank * i, send_counts[i as usize] as usize));
    let b_itr = rng.random_iter::<f32>().take(nsend);
    let msgs: Vec<CPair> = zip(a_itr, b_itr)
        .map(|(a, b)| CPair::new(a, 1.0 / b))
        .collect();

    let mut result = vec![CPair::default(); nrecv];
    let params = All2allvArgs::<i32>::from_counts(&send_counts, &recv_counts);
    all2allv(&msgs, &mut result, &params, &c.comm)?;

    let mut pos: usize = 0;
    for i in 0..c.size {
        for _j in 0..recv_counts[i as usize] {
            ensure_eq!(i * c.rank, result[pos].a);
            result[pos].b *= result[pos].b;
            pos += 1;
        }
    }

    let send_counts2: Vec<usize> =
        recv_counts.iter().map(|x| *x as usize).collect();
    let recv_counts2: Vec<usize> =
        send_counts.iter().map(|x| *x as usize).collect();
    let result2 = all2allv_vec(&result, &send_counts2, &recv_counts2, &c.comm)?;

    ensure_eq!(result2.len(), msgs.len());
    for (r2, m) in zip(result2.iter(), msgs.iter()) {
        ensure_eq!(r2.a, m.a);
        ensure_eq!(r2.b, m.b * m.b);
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
    log_if_error(test_scatter_one(c), c, "SCATTER ONE");
    log_if_error(test_scatter(c), c, "SCATTER");
    log_if_error(test_scatterv(c), c, "SCATTERV");
    log_if_error(test_gather_one(c), c, "GATHER ONE");
    log_if_error(test_gather(c), c, "GATHER");
    log_if_error(test_gatherv(c), c, "GATHERV");
    log_if_error(test_allgather_one(c), c, "ALLGATHER ONE");
    log_if_error(test_allgather(c), c, "ALLGATHER");
    log_if_error(test_allgatherv(c), c, "ALLGATHERV");
    log_if_error(test_all2all(c), c, "ALL2ALL");
    log_if_error(test_all2allv(c), c, "ALL2ALLV");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
