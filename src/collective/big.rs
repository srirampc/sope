//! Collective operations for messages whose per-rank counts exceed
//! `i32::MAX`.
//!
//! Standard MPI calls (`MPI_Scatterv`, `MPI_Gatherv`, `MPI_Alltoall`,
//! `MPI_Alltoallv`) take their counts and displacements as `int`,
//! which limits a single message to at most `i32::MAX` elements. The
//! routines in this module work around that limit by issuing
//! non-blocking point-to-point operations directly:
//!
//! * **Buffered / non-blocking variants** ([`scatterv_big`],
//!   [`gatherv_big`], [`all2all_big`], [`all2allv_big`]) post one
//!   `immediate_send` per destination and one `immediate_receive`
//!   per source inside a single
//!   [`mpi::request::multiple_scope`] and wait for all of them
//!   together. They are typically faster but require a per-rank
//!   receive buffer in addition to the user's output slice.
//!
//! * **Slower scatter-by-scatter variants**
//!   ([`all2allv_via_scatter_big`] and friends) implement the
//!   exchange as `p` successive [`scatterv_big`] calls (one per
//!   source rank). They use less memory but communicate strictly
//!   round by round.
//!
//! The `_vec` flavours allocate the receive buffer; the `_slice`
//! flavours take an explicit pre-allocated output. All inputs are
//! validated against the same helpers used by the regular
//! collectives in [`super`].

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
use mpi::traits::{
    Communicator, CommunicatorCollectives, Destination, Equivalence, Source,
};
use std::iter::zip;

use crate::{All2allvArgs, MCount, util::exc_prefix_sum};

use super::{
    scatter_one, validate_all2all, validate_all2allv, validate_gatherv,
    validate_scatterv,
};

/// `usize`-counts variable-length scatter using non-blocking sends.
///
/// # Description
/// Variant of [`super::scatterv`] for cases where one of the
/// per-rank send sizes does not fit in `i32`. The root rank issues
/// one [`mpi::traits::Destination::immediate_send`] per non-root
/// destination (per its corresponding `send_sizes` entry); the
/// non-root ranks post a single [`mpi::traits::Source::immediate_receive_into`].
/// All requests are awaited inside a single
/// [`mpi::request::multiple_scope`]. The root's own slice is
/// `clone_from_slice`'d directly without a network round-trip.
pub fn scatterv_big<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    let rcv_size = validate_scatterv(s_in, s_out, send_sizes, root, comm)?;
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    let send_offsets: Vec<usize> =
        exc_prefix_sum(send_sizes.iter().cloned(), 1usize);

    // TODO:: send with tag?
    mpi::request::multiple_scope(comm.size() as usize, |scope, coll| {
        if comm.rank() == root {
            for (iu, (s_size, s_offset)) in
                zip(send_sizes.iter(), send_offsets.iter()).enumerate()
            {
                let i = iu as i32;
                if i == root || *s_size == 0 {
                    continue;
                }
                // Do an immediate send to everyone but root
                let st = *s_offset..(*s_offset + *s_size);
                let dest_process = comm.process_at_rank(i);
                let req = dest_process.immediate_send(scope, &s_in[st]);
                coll.add(req);
            }
        } else if rcv_size > 0 {
            // immediate recieve from everyone
            let root_process = comm.process_at_rank(root);
            let req = root_process.immediate_receive_into(scope, &mut s_out[..]);
            coll.add(req);
        }

        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // Sending to self
    if comm.rank() == root && send_sizes[root as usize] > 0 {
        // directly copy to output
        let offset = send_offsets[root as usize];
        let st = offset..(offset + send_sizes[root as usize]);
        s_out.clone_from_slice(&s_in[st]);
    }
    Ok(())
}

/// `scatterv_big` returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`scatterv_big`] that obtains the
/// local receive count via [`scatter_one`] and allocates the output
/// vector for the caller.
pub fn scatterv_big_vec<T>(
    s_in: Option<&[T]>,
    send_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let rcv_size = scatter_one(send_sizes, root, comm)? as usize;
    let mut rcv_vec = vec![T::default(); rcv_size];
    scatterv_big(s_in, &mut rcv_vec, send_sizes, root, comm)?;
    Ok(rcv_vec)
}

/// `usize`-counts variable-length gather using non-blocking ops.
///
/// # Description
/// Variant of [`super::gatherv`] that bypasses the `i32` count
/// limit. The root rank allocates a temporary per-source receive
/// buffer and posts one
/// [`mpi::traits::Source::immediate_receive_into`] per non-root
/// rank; non-root ranks post a single
/// [`mpi::traits::Destination::immediate_send`]. After all
/// requests complete the per-source buffers are concatenated into
/// `s_out` in rank order. The root's own contribution is copied
/// directly without going through the network.
pub fn gatherv_big<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    let snd_size = validate_gatherv(
        s_in,
        s_out.as_ref().map(|x| x.as_ref()),
        recv_sizes,
        root,
        comm,
    )?;
    let s_out = s_out.unwrap_or(&mut []);
    let recv_sizes = recv_sizes.unwrap_or(&[]);

    let mut rcv_buff: Vec<Vec<T>> = if comm.rank() == root {
        recv_sizes
            .iter()
            .map(|rcz| vec![T::default(); *rcz])
            .collect()
    } else {
        vec![]
    };

    // TODO:: send with tag?
    mpi::request::multiple_scope(comm.size() as usize, |scope, coll| {
        if comm.rank() == root {
            //recivers
            for (ui, s_rcv_buf) in rcv_buff.iter_mut().enumerate() {
                if ui as i32 == comm.rank() || s_rcv_buf.is_empty() {
                    continue;
                }
                let snd_process = comm.process_at_rank(ui as i32);
                let req =
                    snd_process.immediate_receive_into(scope, &mut s_rcv_buf[..]);
                coll.add(req);
            }
        } else if snd_size > 0 {
            let root_process = comm.process_at_rank(root);
            let req = root_process.immediate_send(scope, s_in);
            coll.add(req);
        }

        //
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // copy from buffer
    if comm.rank() == root {
        let mut rcv_offset = 0;
        for i in 0..comm.size() {
            let ui = i as usize;
            let r_size = recv_sizes[ui];
            if r_size == 0 {
                continue;
            }
            let r_range = rcv_offset..(rcv_offset + r_size);
            if i != comm.rank() {
                s_out[r_range].clone_from_slice(&rcv_buff[ui]);
            } else {
                // directly copy to output
                s_out[r_range.clone()].clone_from_slice(&s_in[r_range]);
            }
            rcv_offset += recv_sizes[ui];
        }
    }
    Ok(())
}

/// `gatherv_big` returning a `Vec<T>` on root.
///
/// # Description
/// Convenience wrapper around [`gatherv_big`] that allocates the
/// concatenated `Vec<T>` of length `sum(recv_sizes)` on the root
/// rank. Returns `Some(vec)` on root, `None` elsewhere.
pub fn gatherv_big_vec<T>(
    s_in: &[T],
    recv_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(&[]);
        let mut out_vec = vec![T::default(); recv_sizes.iter().sum::<usize>()];
        gatherv_big(s_in, Some(&mut out_vec), Some(recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv_big(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

/// All-to-all (uniform) for messages exceeding the `i32` count limit.
///
/// # Description
/// Each rank splits `a_in` into `comm.size()` equal-size chunks
/// (`a_in.len() / p` per peer) and posts a non-blocking send to
/// every other rank along with a non-blocking receive from every
/// other rank into per-source buffers. After all requests
/// complete, the per-source buffers are written into `a_out` in
/// rank order; the local-to-local chunk is copied directly.
///
/// Validation of `a_in.len()` (multiple of `p`, equal to
/// `a_out.len()`) is performed via [`validate_all2all`].
pub fn all2all_big<T>(
    a_in: &[T],
    a_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2all(a_in, a_out, comm)?;
    // n elements to recieve per processor
    let npp = a_in.len() / (comm.size() as usize);
    let mut rcv_buff: Vec<Vec<T>> =
        (0..comm.size()).map(|_i| vec![T::default(); npp]).collect();

    mpi::request::multiple_scope(2 * comm.size() as usize, |scope, coll| {
        //senders
        for i in 0..comm.size() {
            if i == comm.rank() {
                continue;
            }
            let ui = i as usize;
            let dest_process = comm.process_at_rank(i);
            let snd_offset = ui * npp;
            let s_range = snd_offset..(snd_offset + npp);
            let req = dest_process.immediate_send(scope, &a_in[s_range]);
            coll.add(req);
        }
        for (ui, s_rcv_buf) in rcv_buff.iter_mut().enumerate() {
            let i = ui as i32;
            if i == comm.rank() {
                continue;
            }
            let src_proc = comm.process_at_rank(i);
            let req = src_proc.immediate_receive_into(scope, &mut s_rcv_buf[..]);
            coll.add(req);
        }
        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    for i in 0..comm.size() {
        let ui = i as usize;
        let rcv_offset = ui * npp;
        let r_range = rcv_offset..(rcv_offset + npp);
        if i != comm.rank() {
            a_out[r_range].clone_from_slice(&rcv_buff[ui]);
        } else {
            // directly copy to output
            let snd_offset = ui * npp;
            let s_range = snd_offset..(snd_offset + npp);
            a_out[r_range].clone_from_slice(&a_in[s_range]);
        }
    }

    Ok(())
}

/// `all2all_big` returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2all_big`] that allocates the
/// receive buffer of the same length as `a_in`.
pub fn all2all_big_vec<T>(a_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut recv_buf: Vec<T> = vec![T::default(); a_in.len()];
    all2all_big(a_in, &mut recv_buf, comm)?;
    Ok(recv_buf)
}

/// Variable-length all-to-all for messages exceeding the `i32` count limit.
///
/// # Description
/// Each rank posts a non-blocking send to every peer with a
/// non-zero send count and a non-blocking receive from every peer
/// with a non-zero receive count, using counts and displacements
/// from `args` (converted to `usize`). All requests are awaited
/// inside a single [`mpi::request::multiple_scope`]. After
/// completion, per-source buffers are written into `s_out` at the
/// displacements specified by `args.rcv_disp`; the rank's own
/// (local-to-local) slice is copied directly.
///
/// This is the buffered "fast path" used by [`super::all2allv`] when
/// the per-rank send/receive volumes exceed `i32::MAX`.
pub fn all2allv_big<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
    S: 'static + MCount,
{
    let uargs = args.to_usize();
    let mut rcv_buff: Vec<Vec<T>> = (0..comm.size())
        .map(|i| vec![T::default(); uargs.rcv_cts[i as usize]])
        .collect();
    mpi::request::multiple_scope(2 * comm.size() as usize, |scope, coll| {
        //senders
        for (ui, (snd_offset, snd_count)) in
            zip(uargs.snd_disp.iter(), uargs.snd_cts.iter()).enumerate()
        {
            let i = ui as i32;
            if i == comm.rank() || *snd_count == 0 {
                continue;
            }
            let s_range = *snd_offset..(*snd_offset + *snd_count);
            // Do an immediate send
            let dest_process = comm.process_at_rank(i);
            let req = dest_process.immediate_send(scope, &s_in[s_range]);
            coll.add(req);
        }

        //recivers
        for (ui, (s_rcv_buf, rcv_count)) in
            zip(rcv_buff.iter_mut(), uargs.rcv_cts.iter()).enumerate()
        {
            let i = ui as i32;
            if i == comm.rank() || *rcv_count == 0 {
                continue;
            }
            let snd_process = comm.process_at_rank(i);
            let req =
                snd_process.immediate_receive_into(scope, &mut s_rcv_buf[..]);
            coll.add(req);
        }
        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // copy to output slice
    for i in 0..comm.size() {
        let ui = i as usize;
        if uargs.rcv_cts[ui] == 0 {
            continue;
        }
        let rcv_offset = uargs.rcv_disp[ui];
        let r_range = rcv_offset..(rcv_offset + uargs.rcv_cts[ui]);
        if i != comm.rank() {
            s_out[r_range].clone_from_slice(&rcv_buff[ui]);
        } else {
            // directly copy to output
            let snd_offset = uargs.snd_disp[ui];
            let s_range = snd_offset..(snd_offset + uargs.snd_cts[ui]);
            s_out[r_range].clone_from_slice(&s_in[s_range]);
        }
    }
    Ok(())
}

/// `all2allv_big` from raw send/recv count slices.
///
/// # Description
/// Convenience wrapper around [`all2allv_big`] that builds the
/// [`All2allvArgs`] from the supplied count vectors and validates
/// the slices via [`validate_all2allv`] beforehand.
pub fn all2allv_big_slice<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2allv(s_in, s_out, send_counts, recv_counts, comm)?;
    let params = All2allvArgs::<usize>::from_counts(send_counts, recv_counts);
    all2allv_big(s_in, s_out, &params, comm)
}

/// `all2allv_big` returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2allv_big_slice`] that allocates
/// the receive buffer of length `sum(recv_counts)`.
pub fn all2allv_big_vec<T>(
    s_in: &[T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let recv_total: usize = recv_counts.iter().sum();
    let mut rcv_vec = vec![T::default(); recv_total];
    all2allv_big_slice(s_in, &mut rcv_vec, send_counts, recv_counts, comm)?;
    Ok(rcv_vec)
}

/// All-to-allv as `p` successive [`scatterv_big`] calls.
///
/// # Description
/// Slower (round-by-round) implementation of variable-length
/// all-to-all that does not require allocating per-source receive
/// buffers. For every source rank `i` (`0..p`):
///
/// * if `i == self.rank`, scatters the rank's own `s_in` to every
///   peer using `args.snd_cts`,
/// * otherwise receives the appropriate slice from rank `i`.
///
/// A barrier is issued after each round. Compared to
/// [`all2allv_big`], this trades throughput for a smaller memory
/// footprint, which can be the only viable option when the data
/// volume is extreme.
pub fn all2allv_via_scatter_big<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
    S: 'static + MCount,
{
    let uargs = args.to_usize();
    for (ui, (rcv_start, rcv_size)) in
        zip(args.rcv_disp.iter(), args.rcv_cts.iter()).enumerate()
    {
        let i = ui as i32;
        let rcv_start = rcv_start.to_usize().unwrap();
        let rcv_size = rcv_size.to_usize().unwrap();
        let rcv_s_out = &mut s_out[rcv_start..rcv_start + rcv_size];
        if i == comm.rank() {
            scatterv_big(Some(s_in), rcv_s_out, Some(&uargs.snd_cts), i, comm)?;
        } else {
            scatterv_big(None, rcv_s_out, None, i, comm)?;
        }
        comm.barrier();
    }
    Ok(())
}

/// `all2allv_via_scatter_big` from raw send/recv count slices.
///
/// # Description
/// Convenience wrapper around [`all2allv_via_scatter_big`] that
/// builds the [`All2allvArgs`] from the supplied count vectors and
/// validates the slices via [`validate_all2allv`] beforehand.
pub fn all2allv_via_scatter_big_slice<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2allv(s_in, s_out, send_counts, recv_counts, comm)?;
    let params = All2allvArgs::<usize>::from_counts(send_counts, recv_counts);
    all2allv_via_scatter_big(s_in, s_out, &params, comm)
}

/// `all2allv_via_scatter_big` returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2allv_via_scatter_big_slice`]
/// that allocates the receive buffer of length `sum(recv_counts)`.
pub fn all2allv_via_scatter_big_vec<T>(
    s_in: &[T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let recv_total: usize = recv_counts.iter().sum();
    let mut rcv_vec = vec![T::default(); recv_total];
    all2allv_via_scatter_big_slice(
        s_in,
        &mut rcv_vec,
        send_counts,
        recv_counts,
        comm,
    )?;
    Ok(rcv_vec)
}
