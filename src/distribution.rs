use anyhow::{Ok, Result, bail};
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, Equivalence},
};
use std::collections::VecDeque;
use thiserror::Error;

use crate::{
    collective::{All2allvArgs, all2all_vec, all2allv, allgather_one, scatterv},
    partition::{ArbitDist, Dist},
    reduction::{
        all_of, allreduce_sum, all_same, any_of, exclusive_scan, max_element,
    },
    util::Pair,
};

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid Surplus Queue Status")]
    InvalidSurplusQError,
    #[error("Invalid Root Selection")]
    InvalidRootError,
    #[error("Output Slice is empty")]
    OutSliceLengthError,
    #[error("Input Slice is empty")]
    InSliceLengthError,
    #[error("Surpluses lengths don't match")]
    InvalidSurplusesError,
}

///
/// scatter elements from the process having the maximum number of elements to
/// all the other processes
pub fn distribute_scatter<T>(
    t_in: &[T],
    t_out: &mut [T], // Assuming s_slice has enough size to accept data
    part: &dyn Dist,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
{
    let local_size: usize = t_in.len();
    let (root, _max_size): (i32, usize) = max_element(&local_size, comm);
    if !all_same(&root, comm) {
        bail!(Error::InvalidRootError);
    }
    // if root check input slice not root, check slice has enough space.
    if !all_of(
        if comm.rank() == root {
            local_size == t_in.len()
        } else {
            part.local_size() == t_out.len()
        },
        comm,
    ) {
        bail!(Error::OutSliceLengthError);
    }

    let send_sizes: Vec<i32> = (0..comm.size())
        .map(|px| part.local_size_at(px) as i32)
        .collect();

    scatterv(Some(t_in), t_out, Some(&send_sizes), root, comm)?;
    Ok(())
}

///
/// Trait for distributor  based on sizes
pub trait Distributor<T>
where
    T: Equivalence + Default + Clone,
{
    /// Arguments to all2all
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>>;

    fn distribute(&self, t_in: &[T], t_out: &mut [T]) -> Result<()>;
}

///
/// Distributor implementation that divides the processes into
/// over and under processes  and xfers from the overs to the unders.
pub struct Over2UnderDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
}

impl<'a> Over2UnderDistributor<'a> {
    pub fn new(part: &'a dyn Dist, comm: &'a dyn Communicator) -> Self {
        Self { part, comm }
    }
}

impl<'a, T> Distributor<T> for Over2UnderDistributor<'a>
where
    T: Equivalence + Default + Clone,
{
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>> {
        let p = self.part.comm_size() as usize;
        let rank = self.part.comm_rank() as usize;
        let local_size = t_in.len();
        let in_sizes: Vec<isize> =
            allgather_one(&(local_size as isize), self.comm)?;
        // 1. Partition the processors into senders, recievers and "nothing-to-do"
        //    Compute the over/under for senders/recievers repectively.
        let (over, under): (Vec<isize>, Vec<isize>) =
            self.part.over_under(&in_sizes);
        // ranks of proc that only send
        let senders: Vec<i32> = over
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x > 0 { Some(i as i32) } else { None })
            .collect();
        // ranks of proc that only recive
        let recievers: Vec<i32> = under
            .iter()
            .enumerate()
            .filter_map(|(i, x)| if *x > 0 { Some(i as i32) } else { None })
            .collect();

        let target_size = self.part.block_sizes();
        let mut pm = All2allvArgs::<isize>::new(p);
        let mut sent_offset: isize = 0; // offset starting from which i should send next
        let mut rcvd_offset: isize = 0; // offset starting from which i should rcv next
        // 2. I am rank. Set up the no. of elts. I should send-to/recv-from myself.
        if over[rank] > under[rank] {
            sent_offset = target_size[rank] as isize;
            pm.snd_cts[rank] = target_size[rank] as isize;
            pm.rcv_cts[rank] = target_size[rank] as isize;
            pm.rcv_disp[rank] = 0;
            pm.snd_disp[rank] = 0;
        } else {
            rcvd_offset = in_sizes[rank];
            pm.rcv_cts[rank] = in_sizes[rank];
            pm.snd_cts[rank] = in_sizes[rank];
            pm.rcv_disp[rank] = 0;
            pm.snd_disp[rank] = 0;
        }

        // 3. Set up send/recv counts I should send to/recv frm every one else.
        let mut snd_idx: usize = 0; // processes that only send i.e. are over
        let mut rcv_idx: usize = 0; //  processes that only recv i.e. are under
        let mut sent: isize = 0;
        let mut recvd: isize = 0;
        while snd_idx < senders.len() && rcv_idx < recievers.len() {
            let (snd_rank, rcv_rank) =
                (senders[snd_idx] as usize, recievers[rcv_idx] as usize);
            // I am rank : Setup no. of elts. send-to/recv-from rcv_rank/snd_rank
            // How much ? : xfer the min between what can be send-to/recv-from.
            let xfernow =
                isize::min(over[snd_rank] - sent, under[rcv_rank] - recvd);
            // If I should send, then set up send counts against 'rcv_rank'
            if rank == snd_rank {
                pm.snd_cts[rcv_rank] = xfernow;
                pm.snd_disp[rcv_rank] = sent_offset;
                sent_offset += xfernow;
            }
            // If I should recv, then set up recv counts against 'snd_rank'
            if rank == rcv_rank {
                pm.rcv_cts[snd_rank] = xfernow;
                pm.rcv_disp[snd_rank] = rcvd_offset;
                rcvd_offset += xfernow;
            }
            // Set-up done. Now, update snd_idx, rcv_idx, sent and recvd.
            sent += xfernow;
            recvd += xfernow;
            if sent == over[snd_rank] {
                snd_idx += 1;
                sent = 0;
            }
            if recvd == under[rcv_rank] {
                rcv_idx += 1;
                recvd = 0;
            }
        }
        Ok(pm)
    }

    fn distribute(
        &self,
        t_in: &[T],
        t_out: &mut [T], // Assuming s_slice has enough size to accept data
    ) -> Result<()> {
        // if there's only one process, return a copy
        if self.comm.size() == 1 {
            t_out.clone_from_slice(t_in);
            return Ok(());
        }

        //get local and global size
        let local_size: usize = t_in.len();
        let total_size: usize = allreduce_sum(&local_size, self.comm);
        if total_size == 0 {
            bail!(Error::InSliceLengthError);
        }

        let params = self.cc_args(t_in)?;
        all2allv(t_in, t_out, &params, self.comm)?;
        Ok(())
    }
}

///
/// Distributor implementation that divides the processes into
/// having positive and negative surpluses and xfers from the +ves to the -ves.
pub struct SurplusDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
    send_deficit: Option<bool>,
}

impl<'a> SurplusDistributor<'a> {
    pub fn new(
        part: &'a dyn Dist,
        comm: &'a dyn Communicator,
        send_deficit: Option<bool>,
    ) -> Self {
        Self {
            part,
            comm,
            send_deficit,
        }
    }

    fn surplus_send_counts(
        &self,
        surpluses: &[isize], // negative `surpluses` represents a deficit
    ) -> Result<Vec<usize>> {
        let send_deficit = self.send_deficit.unwrap_or(true);
        let p: i32 = self.comm.size();
        let rank: i32 = self.comm.rank();

        // calculate the send and receive counts by a linear scan over
        // the surpluses, using a queue to keep track of all surpluses
        let mut surpluses = Vec::from(surpluses);
        let mut send_counts = vec![0usize; p as usize];
        let mut fifo: VecDeque<Pair<i32, isize>> = VecDeque::new();
        for i in 0..p as usize {
            let ri = i as i32;
            if surpluses[i] == 0 {
                continue;
            }
            if fifo.is_empty() {
                fifo.push_back(Pair::new(ri, surpluses[i]));
            } else if surpluses[i] > 0 {
                if fifo.front().ok_or(Error::InvalidSurplusQError)?.second > 0 {
                    fifo.push_back(Pair::new(ri, surpluses[i]));
                } else {
                    while surpluses[i] > 0 && !fifo.is_empty() {
                        let fifo_front = fifo
                            .front_mut()
                            .ok_or(Error::InvalidSurplusQError)?;
                        let min: isize =
                            isize::min(surpluses[i], -fifo_front.second);
                        let j: usize = fifo_front.first as usize;
                        surpluses[i] -= min;
                        fifo_front.second += min;
                        if fifo_front.second == 0 {
                            let _ = fifo.pop_front();
                        }
                        // these processors communicate!
                        if rank == ri {
                            send_counts[j] += min as usize;
                        } else if (rank as usize == j) && send_deficit {
                            send_counts[i] += min as usize;
                        }
                    }
                    if surpluses[i] > 0 {
                        fifo.push_back(Pair::new(ri, surpluses[i]))
                    }
                }
            } else if surpluses[i] < 0 {
                if fifo.front().ok_or(Error::InvalidSurplusQError)?.second < 0 {
                    fifo.push_back(Pair::new(ri, surpluses[i]));
                } else {
                    while surpluses[i] < 0 && !fifo.is_empty() {
                        let fifo_front = fifo
                            .front_mut()
                            .ok_or(Error::InvalidSurplusQError)?;
                        let min: isize =
                            isize::min(-surpluses[i], fifo_front.second);
                        let j: usize = fifo_front.first as usize;
                        surpluses[i] += min;
                        fifo_front.second -= min;
                        if fifo_front.second == 0 {
                            let _ = fifo.pop_front();
                        }
                        // these processors communicate!
                        if rank == ri && send_deficit {
                            send_counts[j] += min as usize;
                        } else if rank as usize == j {
                            send_counts[i] += min as usize;
                        }
                    }
                    if surpluses[i] < 0 {
                        fifo.push_back(Pair::new(ri, surpluses[i]));
                    }
                }
            }
        }
        anyhow::ensure!(fifo.is_empty());

        Ok(send_counts)
    }
}

impl<'a, T> Distributor<T> for SurplusDistributor<'a>
where
    T: Equivalence + Default + Clone,
{
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>> {
        let in_sizes: Vec<isize> =
            allgather_one(&(t_in.len() as isize), self.comm)?;
        let surpluses: Vec<isize> = in_sizes
            .iter()
            .enumerate()
            .map(|(i, x)| *x - self.part.local_size_at(i as i32) as isize)
            .collect();
        if !all_of(surpluses.iter().sum::<isize>() == 0, self.comm) {
            bail!(Error::InvalidSurplusesError);
        }
        // use surplus send-pairing to minimize total communication volume
        // get send counts
        let send_counts: Vec<usize> = self.surplus_send_counts(&surpluses)?;
        let recv_counts: Vec<usize> = all2all_vec(&send_counts, self.comm)?;
        // all2allv send/recv counts/displs to balance the surplus
        Ok(All2allvArgs::<isize>::from_counts(
            &send_counts,
            &recv_counts,
        ))
    }

    fn distribute(
        &self,
        t_in: &[T],
        t_out: &mut [T], // Assuming s_slice has enough size to accept data
    ) -> Result<()> {
        if self.comm.size() == 1 {
            t_out.clone_from_slice(t_in);
            return Ok(());
        }
        let local_size = t_in.len();
        let total_size = allreduce_sum(&local_size, self.comm);

        if any_of(total_size == local_size, self.comm) {
            distribute_scatter(t_in, t_out, self.part, self.comm)?;
            return Ok(());
        }
        let params = self.cc_args(t_in)?;
        let surplus: isize =
            local_size as isize - self.part.local_size() as isize;
        // TODO: use all2all or send/recv depending on the maximum number of
        //       paired processes
        if surplus > 0 {
            let n_remain = local_size - surplus as usize;
            let (s_snd, s_rcv) = (&t_in[n_remain..], &mut []);
            all2allv(s_snd, s_rcv, &params, self.comm)?;
            t_out.clone_from_slice(&t_in[..n_remain]);
        } else {
            let n_rcv = params.rcv_cts.iter().map(|x| *x as usize).sum();
            let (s_snd, mut s_rcv) = (&[], vec![T::default(); n_rcv]);
            all2allv(s_snd, &mut s_rcv, &params, self.comm)?;
            t_out.clone_from_slice(t_in);
            t_out[local_size..].clone_from_slice(&s_rcv);
        };
        Ok(())
    }
}

///
/// Distributor implementation that distibutes such that the ordering is retained.
pub struct StableDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
}

impl<'a> StableDistributor<'a> {
    pub fn new(part: &'a dyn Dist, comm: &'a dyn Communicator) -> Self {
        Self { part, comm }
    }
}

impl<'a, T> Distributor<T> for StableDistributor<'a>
where
    T: Equivalence + Default + Clone,
{
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>> {
        let local_size: usize = t_in.len();
        // get prefix sum of size and total size
        let mut start_idx =
            exclusive_scan(&local_size, self.comm, SystemOperation::sum());

        // calculate where to send elements, if there are any elements to send
        let send_counts = if local_size > 0 {
            let mut send_counts: Vec<usize> = vec![0; self.comm.size() as usize];
            let mut target_p: i32 = self.part.owner(start_idx);
            let mut left_to_send: usize = local_size;
            while left_to_send > 0 && target_p < self.comm.size() {
                let nsend =
                    left_to_send.min(self.part.end_at(target_p) - start_idx);
                send_counts[target_p as usize] = nsend;
                left_to_send -= nsend;
                start_idx += nsend;
                target_p += 1;
            }
            send_counts
        } else {
            vec![0; self.comm.size() as usize]
        };

        let recv_counts = all2all_vec(&send_counts, self.comm)?;
        Ok(All2allvArgs::from_counts(&send_counts, &recv_counts))
    }

    fn distribute(
        &self,
        t_in: &[T],
        t_out: &mut [T], // Assuming s_slice has enough size to accept data
    ) -> Result<()> {
        // if there's only one process, return a copy
        if self.comm.size() == 1 {
            t_out.clone_from_slice(t_in);
            return Ok(());
        }

        //get local and global size
        let local_size: usize = t_in.len();
        let total_size: usize = allreduce_sum(&local_size, self.comm);
        if total_size == 0 {
            bail!(Error::InSliceLengthError);
        }

        // one process has all elements -> use scatter instead of all2all
        if any_of(total_size == local_size, self.comm) {
            distribute_scatter(t_in, t_out, self.part, self.comm)?;
            return Ok(());
        }
        let params = self.cc_args(t_in)?;
        all2allv(t_in, t_out, &params, self.comm)?;
        Ok(())
    }
}

///
/// Distributor implementation for distibuting in a arbitrary manner.
pub struct ArbitDistributor<'a> {
    part: ArbitDist,
    comm: &'a dyn Communicator,
}

impl<'a> ArbitDistributor<'a> {
    pub fn new(
        new_local_size: usize,
        comm: &'a dyn Communicator,
    ) -> Result<Self> {
        let sizes = allgather_one(&new_local_size, comm)?;
        let n = allreduce_sum(&new_local_size, comm);
        Ok(Self {
            part: ArbitDist::new(n, comm.size(), comm.rank(), sizes),
            comm,
        })
    }
}

impl<'a, T> Distributor<T> for ArbitDistributor<'a>
where
    T: Equivalence + Default + Clone,
{
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>> {
        let local_size = t_in.len();
        // get prefix sum of size
        let mut prefix =
            exclusive_scan(&local_size, self.comm, SystemOperation::sum());
        let new_local_sizes = self.part.block_sizes();

        // calculate where to send elements
        let mut send_counts = vec![0; self.comm.size() as usize];
        let mut target_p: i32 = 0;
        let mut new_prefix = 0;
        // Find processor for which the prefix sum exceeds mine
        // I have to send to the one preceding that
        while target_p < self.comm.size() - 1 {
            if new_prefix + new_local_sizes[target_p as usize] > prefix {
                break;
            }
            new_prefix += new_local_sizes[target_p as usize];
            target_p += 1;
        }

        //
        let mut left_to_send = local_size;
        while left_to_send > 0 && target_p < self.comm.size() {
            // make the `new` prefix inclusive (is an exlcusive prefix prior)
            new_prefix += new_local_sizes[target_p as usize];
            // send as many elements to the current processor as it needs to fill
            // up, but at most as many as I have left
            let nsend = left_to_send.min(new_prefix - prefix);
            send_counts[target_p as usize] = nsend;
            // update the number of elements i have left (`left_to_send`) and
            // at which global index they start `prefix`
            left_to_send -= nsend;
            prefix += nsend;
            target_p += 1;
        }

        // TODO: all2allv for iterators
        let recv_counts = all2all_vec(&send_counts, self.comm)?;
        Ok(All2allvArgs::from_counts(&send_counts, &recv_counts))
    }

    fn distribute(
        &self,
        t_in: &[T],
        t_out: &mut [T], // Assuming s_slice has enough size to accept data
    ) -> Result<()> {
        // if single process, simply copy to output
        if self.comm.size() == 1 {
            t_out.clone_from_slice(t_in);
            return Ok(());
        }
        let params = self.cc_args(t_in)?;
        all2allv(t_in, t_out, &params, self.comm)?;
        Ok(())
    }
}

pub fn stable_distribute<T>(
    t_in: &[T],
    t_out: &mut [T],
    part: &impl Dist,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
{
    StableDistributor::new(part, comm).distribute(t_in, t_out)
}

// Container stable_distribute(const Container& c, const mxx::comm& comm) {
pub fn stable_distribute_vec<T>(
    tv: &[T],
    part: &impl Dist,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    if comm.size() <= 1 {
        return Ok(Vec::from(tv));
    }

    // allocate and call the slice based implementation
    let mut result: Vec<T> = vec![T::default(); part.local_size()];
    let sdist = StableDistributor::new(part, comm);
    sdist.distribute(tv, &mut result)?;
    Ok(result)
}

pub fn distribute_vec<T>(
    tv: &[T],
    part: &impl Dist,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    if comm.size() == 1 {
        return Ok(Vec::from(tv));
    }

    let mut result: Vec<T> = vec![T::default(); part.local_size()];
    let sdist = StableDistributor::new(part, comm);
    sdist.distribute(tv, &mut result)?;

    Ok(result)
}

pub fn arbit_distribute<T>(
    t_in: &[T],
    t_out: &mut [T], // Assuming t_out has enough size to accept data
    target_local_size: usize,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
{
    let arbit_distr = ArbitDistributor::new(target_local_size, comm)?;
    arbit_distr.distribute(t_in, t_out)?;
    Ok(())
}
