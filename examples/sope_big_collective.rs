use anyhow::{Ok, Result};
use mpi::collective::CommunicatorCollectives;
use rand::RngExt;
use sope::{
    big_collective::{all2allv_big_vec, gatherv_big_vec, scatterv_big_vec},
    collective::{all2all_vec, scatter_one},
    comm::WorldComm,
    cond_debug, cond_println, ensure_eq, gather_debug,
    partition::{Dist, InterleavedDist},
    reduction::{all_of, any_of},
    timer::{CumulativeTimer, SectionTimer},
    util::exc_prefix_sum,
};
use std::iter::zip;

fn init_gatherv_rcv_counts(part: &impl Dist) -> Result<Vec<usize>> {
    let mut rng = rand::rng();
    let mut round_about = |n: usize, fraction: usize| {
        let rx = rng.random::<u64>() as usize;
        n - n / fraction + (rx % (2 * (n / fraction)))
    };
    let c_size = part.comm_size();

    let mut root_size = part.global_size();
    let mut rcv_counts: Vec<usize> = vec![0; c_size as usize];
    for i in 0..(c_size - 1) {
        let s = round_about(part.local_size(), 10);
        rcv_counts[i as usize] = s.min(root_size);
        root_size -= s;
    }
    rcv_counts[c_size as usize - 1] = root_size;
    Ok(rcv_counts)
}

fn test_gatherv_size(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let (rcv_data, rcv_counts) = if c.rank == 0 {
        let rcv_counts = init_gatherv_rcv_counts(&part)?;
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

fn init_scatterv_snd_counts(part: &impl Dist) -> Result<Vec<usize>> {
    let mut rng = rand::rng();
    let mut round_about = |n: usize, fraction: usize| {
        let rx = rng.random::<u64>() as usize;
        n - n / fraction + (rx % (2 * (n / fraction)))
    };
    let c_size = part.comm_size();

    let mut root_size = part.global_size();
    let mut send_counts: Vec<usize> = vec![0; c_size as usize];
    for i in 0..(c_size - 1) {
        let s = round_about(part.local_size(), 10);
        send_counts[i as usize] = s.min(root_size);
        root_size -= s;
    }
    send_counts[c_size as usize - 1] = root_size;
    Ok(send_counts)
}

fn test_scatterv_size(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let (snd_data, send_counts) = if c.rank == 0 {
        let part = InterleavedDist::new(input_size, c.size, c.rank);
        let counts = init_scatterv_snd_counts(&part)?;

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

fn init_a2a_snd_counts(part: &impl Dist) -> Result<Vec<usize>> {
    let mut rng = rand::rng();
    let mut round_about = |n: usize, fraction: usize| {
        let rx = rng.random::<u64>() as usize;
        n - n / fraction + (rx % (2 * (n / fraction)))
    };
    let c_size = part.comm_size();
    let c_rank = part.comm_rank();
    let mut local_size = part.local_size();
    let local_part =
        InterleavedDist::new(local_size, c_size as i32, c_rank as i32);
    let mut send_counts: Vec<usize> = vec![0; c_size as usize];
    for i in 0..(c_size - 1) {
        let s = round_about(local_part.local_size(), 10);
        send_counts[i as usize] = s.min(local_size);
        local_size -= s;
    }
    send_counts[c_size as usize - 1] = local_size;

    Ok(send_counts)
}

fn test_all2allv_size(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let send_counts = init_a2a_snd_counts(&part)?;
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
    s_timer.info_section(&format!("A2A TEST WITH {}", input_size));
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
    let ctimer: CumulativeTimer =  CumulativeTimer::from_comm(&c.comm, ";");
    log_if_error(test_scatterv_size(c, 1024 * 1024), c, "SCATTERV BIG 2^20");
    ctimer.reset();
    log_if_error(
        test_scatterv_size(c, 1024 * 1024 * 1024),
        c,
        "SCATTER BIG 2^30",
    );
    ctimer.add_elapsed();

    log_if_error(test_gatherv_size(c, 1024 * 1024), c, "GATHER BIG 2^20");
    ctimer.reset();
    log_if_error(
        test_gatherv_size(c, 1024 * 1024 * 1024),
        c,
        "GATHER BIG 2^30",
    );
    ctimer.add_elapsed();

    log_if_error(test_all2allv_size(c, 1024 * 1024), c, "A2A BIG 2^20");
    ctimer.reset();
    log_if_error(test_all2allv_size(c, 1024 * 1024 * 1024), c, "A2A BIG 2^30");
    ctimer.add_elapsed();

    if c.size >= 16 {
        // this takes forever ?
        log_if_error(
            test_scatterv_size(c, 1024 * 1024 * 1024 * 16),
            c,
            "SCATTER BIG 2^34",
        );
        log_if_error(
            test_gatherv_size(c, 1024 * 1024 * 1024 * 16),
            c,
            "GATHER BIG 2^34",
        );
        //log_if_error(
        //    test_gatherv_size(c, 1024 * 1024 * 1024 * 8),
        //    c,
        //    "GATHER BIG 2^33",
        //);
        log_if_error(
            test_all2allv_size(c, 1024 * 1024 * 1024 * 16),
            c,
            "A2A BIG 2^34",
        );
    }

    std::thread::sleep(std::time::Duration::from_millis(1000));
    c.comm.barrier();
    ctimer.info_region("TOTAL 2^30");
    cond_println!(c.is_root(); "BIG COLLECTIVES TEST COMPLETED");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
