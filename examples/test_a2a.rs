use anyhow::{Ok, Result};
use num::FromPrimitive;
use rand::{RngExt, rngs::ThreadRng};
use sope::{
    collective::{
        all2all_vec, all2allv_big_vec, all2allv_vec, all2allv_via_scatter_big_vec,
    },
    comm::WorldComm,
    partition::{Dist, InterleavedDist},
    reduction::{allreduce_sum, any_of},
    timer::SectionTimer,
};

struct RandomGen {
    rng: ThreadRng,
}

impl RandomGen {
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }

    /// randomly selct a number that is approximately close to n
    ///  between n + (n/fraction) and n - (n/fraction)
    pub fn rand_approx(&mut self, n: usize, fraction: usize) -> usize {
        //let rx = self.rng.random::<u64>() as usize;
        let rstart = n.min(n - n / fraction);
        let rend = n + (2 * (n / fraction));
        self.rng.random_range(rstart..rend)
    }

    /// randomly generate splits of localt size that is approximately close to
    /// part's local_size with randomly chosen between
    ///   local_counts + (local_counts/fraction) and  
    ///   local_counts - (local_counts/fraction)
    pub fn approx_split_counts(
        &mut self,
        part: &impl Dist,
        fraction: usize,
    ) -> Vec<usize> {
        let comm_size = part.comm_size();
        let mut root_size = part.global_size();
        let mut counts: Vec<usize> = vec![0; comm_size as usize];
        for i in 0..(comm_size - 1) {
            let s = self.rand_approx(part.local_size(), fraction);
            counts[i as usize] = s.min(root_size);
            root_size -= s;
        }
        counts[comm_size as usize - 1] = root_size;
        counts
    }

    pub fn generate<T: FromPrimitive + Default>(
        &mut self,
        glen: usize,
        limit: usize,
    ) -> Vec<T> {
        (0..glen)
            .map(|_r| {
                let rx = self.rng.random::<u64>() as usize;
                T::from_usize(rx % limit).unwrap_or_default()
            })
            .collect()
    }
}

fn test_all2allv(c: &WorldComm, input_size: usize) -> Result<()> {
    let s_timer = SectionTimer::from_comm(&c.comm, ",");
    let part = InterleavedDist::new(input_size, c.size, c.rank);
    let mut rgen = RandomGen::new();
    let send_counts = rgen.approx_split_counts(&part, part.comm_size() as usize);
    let send_size = send_counts.iter().sum();
    let local_els = rgen.generate::<f32>(send_size, 200);
    let rsize = allreduce_sum(&(local_els.len()), &c.comm);
    s_timer.info_section(&format!("A2AV TEST START WITH {}", rsize));
    s_timer.reset();
    let recv_counts = all2all_vec(&send_counts, &c.comm)?;
    let mut results =
        all2allv_vec(&local_els, &send_counts, &recv_counts, &c.comm)?;
    let rsize = allreduce_sum(&(results.len()), &c.comm);
    s_timer.info_section(&format!("A2AV TEST WITH {}", rsize));
    s_timer.reset();
    results.clear();
    results = all2allv_big_vec(&local_els, &send_counts, &recv_counts, &c.comm)?;
    let rsize = allreduce_sum(&(results.len()), &c.comm);
    s_timer.info_section(&format!("A2AV BIG TEST WITH {}", rsize));
    s_timer.reset();
    results.clear();
    results = all2allv_via_scatter_big_vec(
        &local_els,
        &send_counts,
        &recv_counts,
        &c.comm,
    )?;
    let rsize = allreduce_sum(&(results.len()), &c.comm);
    s_timer.info_section(&format!("A2AV VIA SCATTER BIG TEST WITH {}", rsize));

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
    log_if_error(test_all2allv(c, 9724314000), c, "A2AV BIG 9724314000");
}

fn main() {
    let comm_ifx = WorldComm::init();
    run(&comm_ifx);
}
