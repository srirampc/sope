//! Re-distribution of a flat array between MPI processes.
//!
//! Given an input slice that is split across processes in some
//! arbitrary way, the helpers in this module move elements between
//! ranks so that the resulting layout matches a target [`Dist`]
//! partition. Several strategies are provided, each implementing the
//! [`Distributor`] trait:
//!
//! * [`Over2UnderDistributor`] - splits ranks into "over" (more than
//!   target) and "under" (less than target) groups and transfers
//!   elements directly from overs to unders.
//! * [`SurplusDistributor`] - signed-surplus pairing using a FIFO of
//!   pending surpluses/deficits to minimise total communication
//!   volume.
//! * [`StableDistributor`] - re-distributes while preserving the
//!   global ordering of elements.
//! * [`ArbitDistributor`] - re-distributes to an arbitrary per-rank
//!   target size derived at runtime from the requested local size.
//!
//! Convenience free functions [`distribute_scatter`],
//! [`stable_distribute`], [`stable_distribute_vec`],
//! [`distribute_vec`] and [`arbit_distribute`] wrap the most common
//! use-cases.

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

use anyhow::{Ok, Result, bail};
use mpi::{
    collective::SystemOperation,
    traits::{Communicator, Equivalence},
};
use std::collections::VecDeque;
use thiserror::Error;

use crate::{
    All2allvArgs,
    collective::{all2all_vec, all2allv, allgather_one, scatterv},
    partition::{ArbitDist, Dist},
    reduction::{
        all_of, allreduce_sum, all_same, any_of, exclusive_scan, max_element,
    },
    util::Pair,
};

/// Errors produced by the distribution helpers.
#[derive(Error, Debug)]
pub enum Error {
    /// The internal FIFO of surpluses ended up in an inconsistent
    /// state during [`SurplusDistributor`] processing.
    #[error("Invalid Surplus Queue Status")]
    InvalidSurplusQError,
    /// In [`distribute_scatter`], not all processes agreed on the
    /// rank with the maximum number of elements (i.e. there is no
    /// single root to scatter from).
    #[error("Invalid Root Selection")]
    InvalidRootError,
    /// The output slice is shorter than required by the target
    /// distribution.
    #[error("Output Slice is empty")]
    OutSliceLengthError,
    /// No process has any input data, so distribution is impossible.
    #[error("Input Slice is empty")]
    InSliceLengthError,
    /// In [`SurplusDistributor`], the per-rank surpluses do not sum
    /// to zero - they cannot be balanced by re-distribution.
    #[error("Surpluses lengths don't match")]
    InvalidSurplusesError,
}

/// Scatter from the rank with the largest input slice.
///
/// # Description
/// Pick the rank that holds the most elements (using
/// [`max_element`]) and scatter its contents to all the other
/// processes following the layout described by `part`. The total
/// number of elements held across the communicator must equal the
/// chosen root's `t_in.len()`; non-root ranks are expected to pass
/// an output slice of length `part.local_size()`.
///
/// # Arguments
/// * `t_in` - input slice; only meaningful at the root.
/// * `t_out` - output slice on every rank.
/// * `part` - target distribution describing how to split the input.
/// * `comm` - Communicator
///
/// # Errors
/// * `InvalidRootError` if processes disagree on the root.
/// * `OutSliceLengthError` if some rank's output slice is too
///   short.
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

/// Common interface implemented by all distribution strategies.
///
/// # Description
/// A `Distributor` knows how to compute the per-rank send/receive
/// counts and displacements ([`All2allvArgs`]) required to move
/// elements from the input layout to the target layout, and how to
/// run the actual `MPI_Alltoallv` to perform the transfer.
pub trait Distributor<T>
where
    T: Equivalence + Default + Clone,
{
    /// Compute the [`All2allvArgs`] (send/recv counts and
    /// displacements) needed to re-distribute `t_in` according to
    /// this distributor's strategy.
    fn cc_args(&self, t_in: &[T]) -> Result<All2allvArgs<isize>>;

    /// Re-distribute `t_in` into `t_out` across the communicator.
    /// `t_out` must be sized to hold the local portion of the
    /// re-distributed array.
    fn distribute(&self, t_in: &[T], t_out: &mut [T]) -> Result<()>;
}

/// Re-distribute by transferring directly from "over" ranks to
/// "under" ranks.
///
/// # Description
/// Each rank's input size is compared against the target local size
/// from `part`. Ranks that hold more than their target are
/// "senders" (over), ranks that hold less are "receivers" (under).
/// Senders are paired with receivers in rank order and elements are
/// transferred in chunks so that sender `s` is fully drained before
/// moving to the next sender, and receiver `r` is fully filled
/// before moving to the next receiver.
pub struct Over2UnderDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
}

impl<'a> Over2UnderDistributor<'a> {
    /// Construct a new `Over2UnderDistributor` for the target
    /// partition `part` over communicator `comm`.
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

/// Re-distribute using a signed-surplus pairing scheme.
///
/// # Description
/// Each rank computes its `surplus = local_size - target_local_size`
/// (positive = excess to send, negative = deficit to receive). A
/// FIFO of pending surpluses/deficits is built by scanning ranks in
/// order; matching positive and negative entries cancel each other
/// and produce send-counts that minimise the total volume exchanged.
///
/// The optional `send_deficit` flag controls a tie-breaking rule
/// during pairing (whether deficit-side processes also actively
/// "send" an empty notification). It defaults to `true` when `None`
/// is supplied.
pub struct SurplusDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
    send_deficit: Option<bool>,
}

impl<'a> SurplusDistributor<'a> {
    /// Construct a new `SurplusDistributor`.
    ///
    /// # Arguments
    /// * `part` - target partition.
    /// * `comm` - Communicator.
    /// * `send_deficit` - optional override of the deficit-side
    ///   pairing rule (defaults to `true`).
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

    /// Compute per-rank send counts from the per-rank `surpluses`
    /// vector.
    ///
    /// # Description
    /// Linearly scan all ranks and pair surpluses with deficits via
    /// a FIFO. Each pairing produces send entries on both sides
    /// (positive surplus side sends, deficit side optionally sends
    /// a zero-length matching). Returns the per-rank send count
    /// vector for the local rank.
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

/// Re-distribute while preserving the global ordering of elements.
///
/// # Description
/// The local elements at each rank are conceptually concatenated in
/// rank order to form the global array. `StableDistributor` then
/// re-distributes that global array onto the target partition `part`
/// without changing the relative order of elements: element with
/// global index `g` ends up at the rank `part.owner(g)` and at local
/// index `part.local_index(g)`.
///
/// This is the standard choice when the data has a meaningful order
/// (e.g. it is sorted, or contains positional information).
pub struct StableDistributor<'a> {
    part: &'a dyn Dist,
    comm: &'a dyn Communicator,
}

impl<'a> StableDistributor<'a> {
    /// Construct a new `StableDistributor` for target partition
    /// `part` over communicator `comm`.
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

/// Re-distribute to an arbitrary, runtime-derived target layout.
///
/// # Description
/// The caller supplies a new desired local size on each rank. The
/// constructor gathers these sizes via `allgather_one`, builds the
/// implied [`ArbitDist`] target partition, and the resulting
/// distributor re-balances the input so that every rank ends up with
/// exactly `new_local_size` elements. The relative order of elements
/// is preserved (stable re-distribution into an arbitrary layout).
pub struct ArbitDistributor<'a> {
    part: ArbitDist,
    comm: &'a dyn Communicator,
}

impl<'a> ArbitDistributor<'a> {
    /// Construct a new `ArbitDistributor` whose target partition is
    /// derived from the per-rank `new_local_size` values.
    ///
    /// # Arguments
    /// * `new_local_size` - desired number of elements on this
    ///   process after distribution.
    /// * `comm` - Communicator
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

/// Stable re-distribution into a caller-provided output slice.
///
/// # Description
/// Convenience wrapper that constructs a [`StableDistributor`] and
/// runs it. `t_out` must be sized to `part.local_size()`.
///
/// # Arguments
/// * `t_in` - input slice held by the calling process.
/// * `t_out` - output slice on the calling process.
/// * `part` - target partition.
/// * `comm` - Communicator
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

/// Stable re-distribution returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Allocates an output vector of length `part.local_size()` and runs
/// a [`StableDistributor`]. When the communicator has a single
/// process, returns a copy of the input.
///
/// # Arguments
/// * `tv` - input slice held by the calling process.
/// * `part` - target partition.
/// * `comm` - Communicator
///
/// # Returns
/// A `Vec<T>` containing the locally owned slice after distribution.
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

/// Re-distribute into a freshly allocated `Vec<T>` (stable).
///
/// # Description
/// Same behaviour as [`stable_distribute_vec`] - currently
/// implemented on top of [`StableDistributor`]. Provided as a
/// shorthand when callers do not care about the specific strategy
/// and just want each rank to end up with `part.local_size()`
/// elements in global order.
///
/// # Arguments
/// * `tv` - input slice held by the calling process.
/// * `part` - target partition.
/// * `comm` - Communicator
///
/// # Returns
/// A `Vec<T>` containing the locally owned slice after distribution.
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

/// Arbitrary re-distribution to a per-rank target size.
///
/// # Description
/// Convenience wrapper that constructs an [`ArbitDistributor`] from
/// `target_local_size` and runs it. After completion every rank
/// holds exactly `target_local_size` elements in `t_out`.
///
/// # Arguments
/// * `t_in` - input slice held by the calling process.
/// * `t_out` - output slice on the calling process; must be at least
///   `target_local_size` long.
/// * `target_local_size` - desired number of elements at this rank
///   after distribution.
/// * `comm` - Communicator
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
