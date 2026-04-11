use anyhow::{Ok, Result};
use mpi::collective::CommunicatorCollectives;
use rand::{RngExt, rngs::ThreadRng};
use sope::{
    collective::{
        all2all_big_vec, all2all_vec, all2allv_big_vec,
        all2allv_via_scatter_big_vec, gatherv_big_vec, scatter_one,
        scatterv_big_vec,
    },
    comm::WorldComm,
    cond_debug, cond_println, ensure_eq, gather_debug,
    partition::{Dist, InterleavedDist},
    reduction::{all_of, any_of},
    timer::{CumulativeTimer, SectionTimer},
    util::exc_prefix_sum,
};
use std::iter::zip;

struct RandomCounts {
    rng: ThreadRng,
}

impl RandomCounts {
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }

    /// randomly selct a number that is approximately close to n
    ///  between n + (n/fraction) and n - (n/fraction)
    pub fn rand_approx(&mut self, n: usize, fraction: usize) -> usize {
        let rx = self.rng.random::<u64>() as usize;
        n - n / fraction + (rx % (2 * (n / fraction)))
    }

    /// randomly generate count that is approximately close to part's
    /// local size with randomly chosen between
    ///   local_counts + (local_counts/fraction) and  
    ///   local_counts - (local_counts/fraction)
    pub fn approx_split_counts(
        &mut self,
        part: &impl Dist,
        fraction: usize,
    ) -> Vec<usize> {
        let comm_size = part.comm_size();
        let mut target_size = part.global_size();
        let mut counts: Vec<usize> = vec![0; comm_size as usize];
        for i in 0..(comm_size - 1) {
            let s = self.rand_approx(part.local_size(), fraction);
            counts[i as usize] = s.min(target_size);
            target_size -= s;
        }
        counts[comm_size as usize - 1] = target_size;
        counts
    }
}

/// Initialize No. Elements of Recieved guided by part by randomly
fn gatherv_recv_counts(part: &impl Dist) -> Vec<usize> {
    RandomCounts::new().approx_split_counts(part, part.comm_size() as usize)
}

/// Initialize No. Elements of Sent guided by part by randomly
fn scatterv_snd_counts(part: &impl Dist) -> Vec<usize> {
    RandomCounts::new().approx_split_counts(part, part.comm_size() as usize)
}

/// Initialize No. Elements of Sent in all2allv guided by part by randomly
fn a2av_send_counts(part: &impl Dist) -> Vec<usize> {
    let l_part = InterleavedDist::new(
        part.local_size(),
        part.comm_size() as i32,
        part.comm_rank() as i32,
    );

    RandomCounts::new().approx_split_counts(&l_part, part.comm_size() as usize)
}

fn test_gatherv(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let (rcv_data, rcv_counts) = if c.rank == 0 {
        let rcv_counts = gatherv_recv_counts(&part);
        let rcv_total: usize = rcv_counts.iter().sum();
        let rcv_data: Vec<i32> = vec![0; rcv_total];
        (Some(rcv_data), Some(rcv_counts))
    } else {
        (None, None)
    };
    let snd_size: usize = scatter_one(rcv_counts.as_deref(), 0, &c.comm)?;
    let snd_data: Vec<i32> = std::iter::repeat_n(c.rank, snd_size).collect();
    gather_debug!(&c.comm;"G  : {} {}", snd_size, snd_data.len());

    let result = gatherv_big_vec(&snd_data, rcv_counts.as_deref(), 0, &c.comm)?;
    let r_test = c.rank != 0 || result.is_some();
    ensure_eq!(all_of(r_test, &c.comm), true);

    let rcv_test = if c.rank == 0
        && let (Some(rcv_data), Some(rcv_counts)) = (rcv_data, rcv_counts)
    {
        let mut offset: usize = 0;
        let mut rcv_test = vec![false; c.size as usize];
        for i in 0..c.size {
            let i_size = rcv_counts[i as usize];
            let str = offset..(offset + i_size);
            rcv_test[i as usize] = rcv_data[str].iter().all(|x| *x == c.rank);
            offset += i_size;
        }
        log::debug!("RCV TEST : {:?}", rcv_test);
        rcv_test.iter().all(|x| *x)
    } else {
        true
    };
    ensure_eq!(all_of(rcv_test, &c.comm), true);
    s_timer.info_section(&format!("GATHERV TEST WITH {}", input_size));
    Ok(())
}

fn test_scatterv(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let (snd_data, send_counts) = if c.rank == 0 {
        let part = InterleavedDist::new(input_size, c.size, c.rank);
        let counts = scatterv_snd_counts(&part);

        // fill in data
        let n_total: usize = counts.iter().sum();
        let displs: Vec<usize> = exc_prefix_sum(counts.iter().cloned(), 1);
        let mut snd_data: Vec<i32> = vec![0; n_total];
        for (i, (dsp, cts)) in zip(displs.iter(), counts.iter()).enumerate() {
            snd_data[*dsp..(*dsp + *cts)].fill(i as i32);
        }

        (Some(snd_data), Some(counts))
    } else {
        (None, None)
    };
    cond_debug!(
        c.rank == 0;
        "G  : ({:?} {:?})", send_counts, snd_data.as_ref().map(|x| x.len())
    );

    let rcv_data = scatterv_big_vec(
        snd_data.as_deref(),
        send_counts.as_deref(),
        0,
        &c.comm,
    )?;
    let test_val = rcv_data.iter().all(|x| *x == c.rank);
    ensure_eq!(test_val, true);

    s_timer.info_section(&format!("SCATTERV TEST WITH {}", input_size));
    Ok(())
}

fn test_all2all(c: &WorldComm, pp_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    // fill in data
    let mut local_els: Vec<i32> = (0..c.size)
        .flat_map(|i| std::iter::repeat_n(i, pp_size))
        .collect();

    let mut results = all2all_big_vec(&local_els, &c.comm)?;
    let test_val = results.iter().all(|x| *x == c.rank);
    results.dedup();
    local_els.dedup();
    gather_debug!(&c.comm; "R {:?} {:?}", local_els, results);
    if !test_val {
        gather_debug!(&c.comm; "FAILED {:?}", results);
    }

    ensure_eq!(test_val, true);
    s_timer.info_section(&format!("A2A TEST WITH {}", pp_size));
    Ok(())
}

fn test_all2allv(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let send_counts = a2av_send_counts(&part);
    gather_debug!(
        &c.comm; "SND {:?} {}", send_counts, send_counts.iter().sum::<usize>()
    );

    // fill in data
    let mut local_els: Vec<i32> = send_counts
        .iter()
        .enumerate()
        .flat_map(|(i, ncts)| std::iter::repeat_n(i as i32, *ncts))
        .collect();
    ensure_eq!(local_els.len(), part.local_size());

    let recv_counts = all2all_vec(&send_counts, &c.comm)?;
    let mut results =
        all2allv_big_vec(&local_els, &send_counts, &recv_counts, &c.comm)?;
    let test_val = results.iter().all(|x| *x == c.rank);

    results.dedup();
    local_els.dedup();
    gather_debug!(&c.comm; "R {:?} {:?}", local_els, results);
    if !test_val {
        gather_debug!(&c.comm; "FAILED {:?}", results);
    }

    ensure_eq!(test_val, true);
    s_timer.info_section(&format!("A2AV TEST WITH {}", input_size));
    Ok(())
}

fn test_all2allv_via_scatter(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let send_counts = a2av_send_counts(&part);
    gather_debug!(
        &c.comm; "SND {:?} {}", send_counts, send_counts.iter().sum::<usize>()
    );

    // fill in data
    let mut local_els: Vec<i32> = send_counts
        .iter()
        .enumerate()
        .flat_map(|(i, ncts)| std::iter::repeat_n(i as i32, *ncts))
        .collect();
    ensure_eq!(local_els.len(), part.local_size());

    let recv_counts = all2all_vec(&send_counts, &c.comm)?;
    let mut results = all2allv_via_scatter_big_vec(
        &local_els,
        &send_counts,
        &recv_counts,
        &c.comm,
    )?;
    let test_val = results.iter().all(|x| *x == c.rank);

    results.dedup();
    local_els.dedup();
    gather_debug!(&c.comm; "R {:?} {:?}", local_els, results);
    if !test_val {
        gather_debug!(&c.comm; "FAILED {:?}", results);
    }

    ensure_eq!(test_val, true);
    s_timer.info_section(&format!("A2AV VIA SCATTER TEST WITH {}", input_size));
    Ok(())
}

fn log_if_error<T>(ex: Result<T>, c: &WorldComm, tm: &str) {
    if any_of(ex.is_err(), &c.comm) {
        sope::gather_error!(
            &c.comm; "{}",
            ex.map_or_else(|e| e.to_string(), |_r| "".to_string())
        );
    } else {
        sope::cond_info!(c.is_root(); "{} SUCCESSFUL", tm );
    }
}

fn run(c: &WorldComm) {
    let _ = env_logger::try_init();
    log_if_error(test_all2all(c, 2), c, "A2A TEST");
    log_if_error(test_all2all(c, 1024), c, "A2A TEST");

    let ctimer: CumulativeTimer = CumulativeTimer::from_comm(&c.comm, ";");
    log_if_error(test_scatterv(c, 1024 * 1024), c, "SCATTERV BIG 2^20");
    ctimer.reset();
    log_if_error(test_scatterv(c, 1024 * 1024 * 1024), c, "SCATTERV BIG 2^30");
    ctimer.add_elapsed();

    log_if_error(test_gatherv(c, 1024 * 1024), c, "GATHERV BIG 2^20");
    ctimer.reset();
    log_if_error(test_gatherv(c, 1024 * 1024 * 1024), c, "GATHERV BIG 2^30");
    ctimer.add_elapsed();

    log_if_error(test_all2allv(c, 1024 * 1024), c, "A2AV BIG 2^20");
    ctimer.reset();
    log_if_error(test_all2allv(c, 1024 * 1024 * 1024), c, "A2AV BIG 2^30");
    ctimer.add_elapsed();

    log_if_error(
        test_all2allv_via_scatter(c, 1024 * 1024),
        c,
        "A2AV VIA SCTTER BIG 2^20",
    );
    ctimer.reset();
    log_if_error(
        test_all2allv_via_scatter(c, 1024 * 1024 * 1024),
        c,
        "A2AV VIA SCTTER BIG 2^30",
    );
    ctimer.add_elapsed();

    if c.size >= 16 {
        // this takes forever ?
        log_if_error(
            test_scatterv(c, 1024 * 1024 * 1024 * 16),
            c,
            "SCATTERV BIG 2^34",
        );
        log_if_error(
            test_gatherv(c, 1024 * 1024 * 1024 * 16),
            c,
            "GATHERV BIG 2^34",
        );
        //log_if_error(
        //    test_gatherv_size(c, 1024 * 1024 * 1024 * 8),
        //    c,
        //    "GATHER BIG 2^33",
        //);
        log_if_error(
            test_all2allv(c, 1024 * 1024 * 1024 * 16),
            c,
            "A2AV BIG 2^34",
        );
        log_if_error(
            test_all2allv_via_scatter(c, 1024 * 1024 * 1024 * 16),
            c,
            "A2AV VIA SCATTER BIG 2^34",
        );
    }

    std::thread::sleep(std::time::Duration::from_millis(2000));
    c.comm.barrier();
    ctimer.info_region("TOTAL 2^30");
    cond_println!(c.is_root(); "BIG COLLECTIVES TEST COMPLETED");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
