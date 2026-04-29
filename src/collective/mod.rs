//! MPI collective operations on Rust data structures.
//!
//! This module wraps the underlying `rsmpi` collectives (`scatter`,
//! `scatterv`, `gather`, `gatherv`, `allgather`, `allgatherv`,
//! `all2all`, `all2allv`) with versions that:
//!
//! * accept Rust slices and `Vec`s instead of raw datatype handles,
//! * validate the input/output sizes across all ranks before
//!   issuing the MPI call,
//! * automatically fall back to the [`big`] sub-module when any
//!   rank's send/receive volume exceeds `i32::MAX` (the count limit
//!   of the underlying MPI calls).
//!
//! The single-element variants (`scatter_one`, `gather_one`,
//! `allgather_one`) and the `*_vec` flavours allocate the receive
//! buffer on behalf of the caller. Validation helpers
//! ([`validate_all2all`], [`validate_all2allv`],
//! [`validate_scatterv`], [`validate_gatherv`]) check the sizes
//! consistently using collective predicates (`all_of`, `any_of`,
//! `all_same`) before kicking off the actual transfer.

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
    datatype::{Partition, PartitionMut},
    traits::{Communicator, CommunicatorCollectives, Equivalence, Root},
};
use num::ToPrimitive;
use std::iter::zip;
use thiserror::Error;

use crate::{
    All2allvArgs, MCount,
    reduction::{all_of, all_same, allreduce, any_of},
    util::exc_prefix_sum_iter,
};

/// Errors produced by the collective wrappers.
#[derive(Error, Debug)]
pub enum Error {
    /// The output slice was shorter than the validated requirement.
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    /// The input slice did not satisfy a per-collective constraint
    /// (e.g. wrong length, empty when expected non-empty, etc.).
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
}

/// Validate inputs of an all-to-all (uniform) call.
///
/// # Description
/// Asserts (collectively) that every rank's `a_in` is non-empty and
/// of length divisible by the communicator size, and that
/// `a_out.len() == a_in.len()` everywhere.
///
/// # Arguments
/// * `a_in` - per-rank input slice for all2all.
/// * `a_out` - per-rank mutable output slice for all2all.
/// * `comm` - Communicator
///
/// # Errors
/// * [`Error::InSliceError`] when the input length is not a multiple
///   of `p`.
/// * [`Error::OutSliceLengthError`] when the output length differs
///   from the input.
pub fn validate_all2all<T>(
    a_in: &[T],
    a_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    if !all_of(
        !a_in.is_empty() && a_in.len().is_multiple_of(comm.size() as usize),
        comm,
    ) {
        bail!(Error::InSliceError(
            "all2all input len should be multiple of p.".to_string()
        ));
    }
    if !all_of(a_out.len() == a_in.len(), comm) {
        bail!(Error::OutSliceLengthError(a_in.len(), a_out.len()));
    }
    Ok(())
}

/// Validate inputs of an all-to-allv (variable-length) call.
///
/// # Description
/// Checks that the input slice covers `sum(send_counts)` elements and
/// that the output slice has space for `sum(recv_counts)` elements
/// on every rank. When the totals are zero the corresponding slices
/// must be empty.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `send_counts` - number of elements to send from 
///    this process to each of the other processes.
/// * `recv_counts` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn validate_all2allv<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    let send_total: usize = send_counts.iter().sum();
    if !all_of(
        if send_total == 0 {
            s_in.is_empty()
        } else {
            s_in.len() >= send_total
        },
        comm,
    ) {
        bail!(Error::InSliceError(
            "all2allv input slice length should be sum of send counts"
                .to_string()
        ));
    }
    let recv_total: usize = recv_counts.iter().sum();
    if !all_of(
        if recv_total == 0 {
            s_out.is_empty()
        } else {
            recv_total <= s_out.len()
        },
        comm,
    ) {
        bail!(Error::OutSliceLengthError(recv_total, s_out.len()));
    }
    Ok(())
}

/// Validate inputs of a scatterv call.
///
/// # Description
/// Verifies that on the `root` rank the input is non-empty, that
/// `send_sizes` has at least `comm.size()` entries, that `s_in` is
/// large enough to cover `sum(send_sizes)`, and that every rank's
/// output slice can hold the receive count obtained via
/// [`scatter_one`].
///
/// # Arguments
/// * `s_in` - input slice for scatterv at root, None everywhere else.
/// * `s_out` - per-rank mutable output slice for scatterv.
/// * `send_sizes` - number of elements to send from 
///    root process to each of the other processes.
/// * `root` - rank of the root process to scatter from.
/// * `comm` - Communicator
///
/// # Returns
/// The receive count delivered to the calling rank (so that callers
/// do not need to recompute it).
pub fn validate_scatterv<T, S>(
    s_in: Option<&[T]>,
    s_out: &[T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[S]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<S>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    let s_total = send_sizes
        .iter()
        .map(|x| x.to_usize().unwrap_or_default())
        .sum();
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && send_sizes.len() >= comm.size() as usize
            && s_in.len() >= s_total,
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatterv input size @ root should be >= sum of send_sizes"
                .to_string()
        ))
    }
    let rcv_size = scatter_one(Some(send_sizes), root, comm)?;
    let o_size: usize = rcv_size.to_usize().unwrap_or_default();
    if !all_of(
        if o_size == 0 {
            s_out.is_empty()
        } else {
            s_out.len() >= o_size
        },
        comm,
    ) {
        bail!(Error::OutSliceLengthError(o_size, s_out.len()));
    }
    Ok(rcv_size)
}

/// Validate inputs of a gatherv call.
///
/// # Description
/// First scatters `recv_sizes` from `root` so that every rank knows
/// how many elements it is expected to send, then verifies the
/// per-rank `s_in` length. On `root`, also asserts that `s_out` is
/// large enough to hold `sum(recv_sizes)`.
///
/// # Arguments
/// * `s_in` - pre-rank input slice for gatherv.
/// * `s_out` - mutable output slice for scatterv at root, None everywhere else.
/// * `recv_sizes` - number of elements to recieve from 
///    each of the other processes by the root.
/// * `root` - rank of the root process to gather to.
/// * `comm` - Communicator
///
/// # Returns
/// The send count expected from the calling rank.
pub fn validate_gatherv<T, S>(
    s_in: &[T],
    s_out: Option<&[T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[S]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<S>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let snd_size = scatter_one(recv_sizes, root, comm)?;
    let snd_usize = snd_size.to_usize().unwrap_or_default();
    let i_len = s_in.len();
    if !all_of(
        if snd_usize == 0 {
            s_in.is_empty()
        } else {
            i_len >= snd_usize
        },
        comm,
    ) {
        bail!(Error::InSliceError(format!(
            "gather input size should be atleast recv_sizes @ root: R({snd_usize}) != IN({i_len})."
        )))
    }
    let s_out = s_out.unwrap_or(&[]);
    let recv_sizes = recv_sizes.unwrap_or(&[]);
    let exp_osize = recv_sizes
        .iter()
        .map(|x| x.to_usize().unwrap_or_default())
        .sum();
    if !any_of(
        comm.rank() == root && exp_osize > 0 && exp_osize <= s_out.len(),
        comm,
    ) {
        bail!(Error::OutSliceLengthError(exp_osize, s_out.len()));
    }
    Ok(snd_size)
}

/// Scatter one element to every process.
///
/// # Description
/// On the `root` rank, `s_in` must contain at least `comm.size()`
/// elements; element `i` is sent to rank `i`. Other ranks pass
/// `None`. Each rank receives exactly one `T`.
///
/// # Returns
/// The element delivered to the calling rank.
///
/// # Errors
/// [`Error::InSliceError`] when the root's input is too short.
pub fn scatter_one<T>(
    s_in: Option<&[T]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<T>
where
    T: Equivalence + Default + Clone,
{
    let s_in = s_in.unwrap_or(&[]);
    if !any_of(
        comm.rank() == root && s_in.len() >= comm.size() as usize,
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatter_one input @ root should be >= p.".to_string()
        ));
    }
    let mut rt = T::default();
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        root_process.scatter_into_root(s_in, &mut rt);
    } else {
        root_process.scatter_into(&mut rt);
    }
    Ok(rt)
}

/// Gather one element from every process to the root.
///
/// # Description
/// Each rank contributes a single `T`. On the `root` rank a
/// `Vec<T>` of length `comm.size()` is returned (rank `i`'s
/// contribution at index `i`); other ranks get `None`.
pub fn gather_one<T>(
    s_in: &T,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let mut rcv_vec = vec![T::default(); comm.size() as usize];
        root_process.gather_into_root(s_in, &mut rcv_vec);
        Ok(Some(rcv_vec))
    } else {
        root_process.gather_into(s_in);
        Ok(None)
    }
}

mod big;
pub use big::{
    all2all_big_vec, all2allv_big, all2allv_big_slice, all2allv_big_vec,
    all2allv_via_scatter_big, all2allv_via_scatter_big_slice,
    all2allv_via_scatter_big_vec, gatherv_big, gatherv_big_vec, scatterv_big,
    scatterv_big_vec,
};

/// Scatter equal-sized chunks of a slice from `root` to every rank.
///
/// # Description
/// On `root`, `s_in` must be non-empty and divisible by
/// `comm.size()`; `s_in` is split into `comm.size()` equal chunks
/// and chunk `i` is delivered to rank `i`. Other ranks pass `None`
/// for the input and a correctly sized output slice.
///
/// # Errors
/// * [`Error::InSliceError`] when the root's input is empty or its
///   size is not a multiple of the communicator size.
/// * [`Error::OutSliceLengthError`] when some rank's output slice is
///   the wrong size.
pub fn scatter<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let s_in = s_in.unwrap_or(&[]);
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && s_in.len().is_multiple_of(comm.size() as usize),
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatter input size @ root should be non-zero and a multipe of p."
                .to_string()
        ))
    }
    let mut exp_size = if comm.rank() == root {
        s_in.len() / comm.size() as usize
    } else {
        0
    };

    if !all_same(
        &(if comm.rank() == root {
            exp_size
        } else {
            s_out.len()
        }),
        comm,
    ) {
        let root_process = comm.process_at_rank(root);
        root_process.broadcast_into(&mut exp_size);
        bail!(Error::OutSliceLengthError(exp_size, s_out.len()));
    }

    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        root_process.scatter_into_root(s_in, s_out);
    } else {
        root_process.scatter_into(s_out);
    }
    Ok(())
}

/// Scatter equal-sized chunks and return the local chunk as a `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`scatter`] that broadcasts the chunk
/// size from `root`, allocates the output buffer, and returns it.
pub fn scatter_vec<T>(
    s_in: Option<&[T]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    // TODO:: handle large sizes
    let s_in = s_in.unwrap_or(&[]);
    let mut exp_size = if comm.rank() == root {
        s_in.len() / comm.size() as usize
    } else {
        0
    };
    comm.process_at_rank(root).broadcast_into(&mut exp_size);
    let mut v_out: Vec<T> = vec![T::default(); exp_size];
    scatter(Some(s_in), &mut v_out, root, comm)?;
    Ok(v_out)
}

/// Variable-length scatter from `root` to every rank.
///
/// # Description
/// On the root rank, `s_in` must hold `sum(send_sizes)` elements
/// and `send_sizes[i]` is the count delivered to rank `i`.
/// Displacements are computed as the exclusive prefix sum of
/// `send_sizes`. Other ranks pass `None` for the input and the
/// sizes vector and a sufficiently sized output slice. Validation is
/// done via [`validate_scatterv`].
pub fn scatterv<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    validate_scatterv(s_in, s_out, send_sizes, root, comm)?;
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let displs: Vec<i32> =
            exc_prefix_sum_iter(send_sizes.iter(), 1).collect();
        let partition = Partition::new(s_in, send_sizes, displs);
        root_process.scatter_varcount_into_root(&partition, s_out);
    } else {
        root_process.scatter_varcount_into(s_out);
    }
    Ok(())
}

/// Variable-length scatter returning the local chunk as a `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`scatterv`]: the local receive count
/// is obtained via [`scatter_one`] from `send_sizes` and the
/// output `Vec<T>` is allocated for the caller.
pub fn scatterv_vec<T>(
    s_in: Option<&[T]>,
    send_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let rcv_size = scatter_one(send_sizes, root, comm)? as usize;
    let mut rcv_vec = vec![T::default(); rcv_size];
    scatterv(s_in, &mut rcv_vec, send_sizes, root, comm)?;
    Ok(rcv_vec)
}

/// Gather equal-sized chunks from every rank to `root`.
///
/// # Description
/// Every rank contributes a slice `s_in` of identical length. On
/// `root`, `s_out` must have space for `s_in.len() * comm.size()`
/// elements; on non-root ranks `s_out` is ignored.
///
/// # Errors
/// * [`Error::InSliceError`] when the per-rank input lengths
///   differ.
/// * [`Error::OutSliceLengthError`] when the root's output is too
///   short.
pub fn gather<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let s_out = s_out.unwrap_or(&mut []);
    if !all_same(&(s_in.len()), comm) {
        bail!(Error::InSliceError(
            "gather input sizes should be same across all processors".to_string()
        ))
    }
    let root_process = comm.process_at_rank(root);
    let exp_size = s_in.len() * comm.size() as usize;
    if !any_of(comm.rank() == root && exp_size <= s_out.len(), comm) {
        bail!(Error::OutSliceLengthError(exp_size, s_out.len()));
    }

    if comm.rank() == root {
        root_process.gather_into_root(s_in, s_out);
    } else {
        root_process.gather_into(s_in);
    }
    Ok(())
}

/// Gather equal-sized chunks and return the gathered `Vec<T>` on root.
///
/// # Description
/// Convenience wrapper around [`gather`] that allocates the receive
/// buffer on the root rank. Returns `Some(vec)` on root, `None`
/// elsewhere.
pub fn gather_vec<T>(
    s_in: &[T],
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let mut out_vec = vec![T::default(); s_in.len() * comm.size() as usize];
        gather(s_in, Some(&mut out_vec), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gather(s_in, None, root, comm)?;
        Ok(None)
    }
}

/// Variable-length gather to `root`.
///
/// # Description
/// Each rank sends `s_in` of arbitrary length; on `root`,
/// `recv_sizes[i]` is the count expected from rank `i` and `s_out`
/// must hold their sum. Displacements are computed as the
/// exclusive prefix sum of `recv_sizes`. Validation is performed
/// via [`validate_gatherv`].
pub fn gatherv<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    validate_gatherv(
        s_in,
        s_out.as_ref().map(|x| x.as_ref()),
        recv_sizes,
        root,
        comm,
    )?;

    let s_out = s_out.unwrap_or(&mut []);
    let recv_sizes = recv_sizes.unwrap_or(&[]);
    let exp_osize = recv_sizes.iter().sum::<i32>() as usize;
    if !any_of(
        comm.rank() == root && exp_osize > 0 && exp_osize <= s_out.len(),
        comm,
    ) {
        bail!(Error::OutSliceLengthError(exp_osize, s_out.len()));
    }

    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let displs: Vec<i32> =
            exc_prefix_sum_iter(recv_sizes.iter(), 1).collect();
        let mut partition = PartitionMut::new(s_out, recv_sizes, displs);
        root_process.gather_varcount_into_root(s_in, &mut partition);
    } else if !s_in.is_empty() {
        root_process.gather_varcount_into(s_in);
    }
    Ok(())
}

/// Variable-length gather returning a `Vec<T>` on root.
///
/// # Description
/// Convenience wrapper around [`gatherv`] that allocates the
/// concatenated `Vec<T>` on `root`. Returns `Some(vec)` on root,
/// `None` elsewhere.
pub fn gatherv_vec<T>(
    s_in: &[T],
    recv_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(&[]);
        let mut out_vec =
            vec![T::default(); recv_sizes.iter().sum::<i32>() as usize];
        gatherv(s_in, Some(&mut out_vec), Some(recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

/// Variable-length gather without a pre-known per-rank size vector.
///
/// # Description
/// First runs a [`gather_one`] to collect the per-rank input lengths
/// at `root`, then performs a [`gatherv`] using those lengths. Useful
/// when the caller does not know the receive sizes in advance.
pub fn gatherv_full_vec<T>(
    s_in: &[T],
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    let ilen: i32 = s_in.len() as i32;
    let recv_sizes = gather_one(&ilen, root, comm)?;
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(vec![]);
        let mut out_vec =
            vec![T::default(); recv_sizes.iter().sum::<i32>() as usize];
        gatherv(s_in, Some(&mut out_vec), Some(&recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

/// Gather per-rank `String` values to `root`.
///
/// # Description
/// Each rank contributes a `String` (encoded as UTF-8 bytes) of
/// arbitrary length. The byte lengths are first gathered with
/// [`gather_one`] and then the bytes themselves with [`gatherv_vec`].
/// On `root`, the resulting bytes are split back into a
/// `Vec<String>` in rank order; empty strings are filtered out.
///
/// Returns `Some(strings)` on `root`, `None` on every other rank.
/// This is the primitive used by the gather-style logging macros in
/// the [`crate::log`] module.
pub fn gather_strings(
    x: String,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<String>>> {
    let lengths: Option<Vec<i32>> = gather_one(&(x.len() as i32), root, comm)?;
    let g_in =
        gatherv_vec(x.as_bytes(), lengths.as_ref().map(|x| &x[..]), root, comm)?;
    if let (Some(sv), Some(lengths)) = (g_in, lengths) {
        let displs: Vec<i32> = exc_prefix_sum_iter(lengths.iter(), 1).collect();
        let svec: Vec<String> = zip(displs.iter(), lengths.iter())
            .map(|(s, l)| {
                let (ts, tl) = (*s as usize, *l as usize);
                String::from_utf8(sv[ts..(ts + tl)].to_vec()).unwrap_or_default()
            })
            .filter(|x| !x.is_empty())
            .collect();
        Ok(Some(svec))
    } else {
        Ok(None)
    }
}

/// All-gather one element from every process to every process.
///
/// # Description
/// Each rank contributes a single `T`. Returns a `Vec<T>` of length
/// `comm.size()` on every rank with rank `i`'s contribution at
/// index `i`.
///
/// # Arguments
/// * `g_in` - per-rank reference to input for allgather.
/// * `comm` - Communicator
pub fn allgather_one<T>(g_in: &T, comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut g_out = vec![T::default(); comm.size() as usize];
    comm.all_gather_into(g_in, &mut g_out);
    Ok(g_out)
}

/// All-gather equal-sized chunks from every rank to every rank.
///
/// # Description
/// Every rank's `g_in` must have the same length; `g_out` must hold
/// `g_in.len() * comm.size()` elements on every rank. Output layout
/// matches [`gather`] but the result is replicated everywhere.
///
/// # Arguments
/// * `g_in` - per-rank input slice for allgather.
/// * `g_out` - per-rank mutable output slice for allgather.
/// * `comm` - Communicator
pub fn allgather<T>(
    g_in: &[T],
    g_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    if !all_same(&g_in.len(), comm) {
        bail!(Error::InSliceError(
            "allgather input size should be same across all procs.".to_string()
        ));
    }
    let exp_len = g_in.len() * comm.size() as usize;
    if !all_of(g_out.len() == exp_len, comm) {
        bail!(Error::OutSliceLengthError(exp_len, g_out.len()));
    }
    comm.all_gather_into(g_in, g_out);
    Ok(())
}

/// All-gather equal-sized chunks returning a `Vec<T>` on every rank.
///
/// # Description
/// Convenience wrapper around [`allgather`] that allocates the
/// output buffer of length `g_in.len() * comm.size()`.
///
/// # Arguments
/// * `g_in` - per-rank input slice for allgather.
/// * `comm` - Communicator
pub fn allgather_vec<T>(g_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut g_out = vec![T::default(); g_in.len() * comm.size() as usize];
    allgather(g_in, &mut g_out, comm)?;
    Ok(g_out)
}

/// Variable-length all-gather across every rank.
///
/// # Description
/// Each rank contributes `g_in` (its length must be
/// `recv_sizes[rank]`); `g_out` must hold `sum(recv_sizes)` elements
/// on every rank. Displacements are computed as the exclusive
/// prefix sum of `recv_sizes`.
///
/// # Arguments
/// * `g_in` - per-rank input slice for allgatherv.
/// * `g_out` - per-rank mutable output slice for allgatherv.
/// * `recv_sizes` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn allgatherv<T>(
    g_in: &[T],
    g_out: &mut [T],
    recv_sizes: &[i32],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let r_len = recv_sizes[comm.rank() as usize] as usize;
    if !all_of(
        if r_len == 0 {
            g_in.is_empty()
        } else {
            g_in.len() >= r_len
        },
        comm,
    ) {
        bail!(Error::InSliceError(
            "gatherv input size should be at least the total recieve sizes."
                .to_string()
        ))
    }

    let exp_len = recv_sizes.iter().sum::<i32>() as usize;
    if !all_of(g_out.len() >= exp_len, comm) {
        bail!(Error::OutSliceLengthError(exp_len, g_out.len()));
    }

    let displs: Vec<i32> = exc_prefix_sum_iter(recv_sizes.iter(), 1).collect();
    let mut partition = PartitionMut::new(g_out, recv_sizes, displs);
    comm.all_gather_varcount_into(g_in, &mut partition);
    Ok(())
}

/// Variable-length all-gather returning a `Vec<T>` on every rank.
///
/// # Description
/// Convenience wrapper around [`allgatherv`] that allocates the
/// receive buffer.
///
/// # Arguments
/// * `g_in` - per-rank input slice for allgatherv.
/// * `recv_sizes` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn allgatherv_vec<T>(
    g_in: &[T],
    recv_sizes: &[i32],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let out_len = recv_sizes.iter().sum::<i32>() as usize;
    let mut g_out = vec![T::default(); out_len];
    allgatherv(g_in, &mut g_out, recv_sizes, comm)?;
    Ok(g_out)
}

/// Variable-length all-gather without a pre-known size vector.
///
/// # Description
/// First runs an [`allgather_one`] to collect every rank's input
/// length, then performs an [`allgatherv_vec`]. Useful when the
/// caller does not know the per-rank sizes in advance.
///
/// # Arguments
/// * `s_in` - per-rank input slice for allgatherv.
/// * `comm` - Communicator
pub fn allgatherv_full_vec<T>(
    s_in: &[T],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let ilen: i32 = s_in.len() as i32;
    let recv_sizes = allgather_one(&ilen, comm)?;
    allgatherv_vec(s_in, &recv_sizes, comm)
}

/// All-to-all (uniform) data exchange.
///
/// # Description
/// `a_in` must be of length divisible by `comm.size()`; chunk `i`
/// is sent to rank `i` and the corresponding chunks from every
/// rank land in `a_out`. Validation is performed via
/// [`validate_all2all`].
///
/// # Arguments
/// * `a_in` - per-rank input slice for all2all.
/// * `a_out` - per-rank mutable output slice for all2all.
/// * `comm` - Communicator
pub fn all2all<T>(
    a_in: &[T],
    a_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    validate_all2all(a_in, a_out, comm)?;
    comm.all_to_all_into(a_in, a_out);
    Ok(())
}

/// All-to-all (uniform) returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2all`] that allocates the receive
/// buffer of the same length as `a_in`.
///
/// # Arguments
/// * `a_in` - per-rank input slice for all2all.
/// * `comm` - Communicator
pub fn all2all_vec<T>(a_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut recv_buf: Vec<T> = vec![T::default(); a_in.len()];
    comm.all_to_all_into(a_in, &mut recv_buf);
    Ok(recv_buf)
}

/// Direct (i32-counts) implementation of [`all2allv`] using `rsmpi`'s
/// `Partition` / `PartitionMut`.
///
/// # Description
/// Internal helper used by [`all2allv`] when the per-rank totals
/// fit inside an `i32`. Counts and displacements from `args` are
/// truncated to `i32` via [`All2allvArgs::to_i32`] before being
/// passed to `comm.all_to_all_varcount_into`.
fn all2allv_<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let iargs = args.to_i32();
    let send_part = Partition::new(s_in, &iargs.snd_cts[..], &iargs.snd_disp[..]);
    let mut rcv_part =
        PartitionMut::new(s_out, &iargs.rcv_cts[..], &iargs.rcv_disp[..]);
    comm.all_to_all_varcount_into(&send_part, &mut rcv_part);
    Ok(())
}

/// Variable-length all-to-all transfer.
///
/// # Description
/// Send/receive counts and displacements are read from `args`. If
/// the maximum per-rank send or receive volume exceeds `i32::MAX`,
/// dispatches to [`big::all2allv_big`] (which uses non-blocking
/// point-to-point messages); otherwise dispatches to the direct
/// MPI varcount call via [`all2allv_`].
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `args` - [`All2allvArgs`]  object encapsulating send and receive counts.
/// * `comm` - Communicator
pub fn all2allv<T, S>(
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
    let total_send = uargs.snd_cts.iter().sum::<usize>();
    let total_rcv = uargs.rcv_cts.iter().sum::<usize>();
    let local_max = total_send.max(total_rcv);
    let g_max = allreduce(&local_max, comm, SystemOperation::max());
    //  Handle large size
    if g_max > i32::MAX as usize {
        big::all2allv_big(s_in, s_out, args, comm)
    } else {
        all2allv_(s_in, s_out, args, comm)
    }
}

/// Variable-length all-to-all from raw send/recv count slices.
///
/// # Description
/// Convenience wrapper around [`all2allv`] that builds the
/// [`All2allvArgs`] from the supplied count vectors (displacements
/// are derived as exclusive prefix sums) and validates the slices
/// via [`validate_all2allv`] beforehand.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `send_counts` - number of elements to send from 
///    this process to each of the other processes.
/// * `recv_counts` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn all2allv_slice<T>(
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
    all2allv(s_in, s_out, &params, comm)
}

/// Variable-length all-to-all returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2allv_slice`] that allocates the
/// receive buffer of length `sum(recv_counts)`.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `send_counts` - number of elements to send from 
///    this process to each of the other processes.
/// * `recv_counts` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn all2allv_vec<T>(
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
    all2allv_slice(s_in, &mut rcv_vec, send_counts, recv_counts, comm)?;
    Ok(rcv_vec)
}

/// `i32`-counts implementation of [`all2allv_via_scatter`].
///
/// # Description
/// Performs `comm.size()` consecutive scatterv calls (one per
/// originating rank), each writing into the appropriate slice of
/// `s_out`. A barrier is issued between rounds. This is the
/// fallback used when `MPI_Alltoallv` is not desired (or when
/// `args` is being staged through scatterv-only primitives).
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `args` - [`All2allvArgs`]  object encapsulating send and receive counts.
/// * `comm` - Communicator
fn all2allv_via_scatter_<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let iargs = args.to_i32();
    for i in 0..comm.size() {
        let rcv_start = iargs.rcv_disp[i as usize].to_usize().unwrap();
        let rcv_size = iargs.rcv_cts[i as usize].to_usize().unwrap();
        let rcv_s_out = &mut s_out[rcv_start..rcv_start + rcv_size];
        if i == comm.rank() {
            scatterv(Some(s_in), rcv_s_out, Some(&iargs.snd_cts), i, comm)?;
        } else {
            scatterv(None, rcv_s_out, None, i, comm)?;
        }
        comm.barrier();
    }
    Ok(())
}

/// All-to-allv implemented as `p` successive scatterv calls.
///
/// # Description
/// Alternative to [`all2allv`] that issues `comm.size()` rounds of
/// [`scatterv`] (one per source rank) instead of a single
/// `MPI_Alltoallv`. Useful when a) the underlying MPI implementation
/// has weak `Alltoallv` performance for the given sizes, or b)
/// callers want a simpler exchange that can be debugged round by
/// round. Currently `todo!`s when total volume exceeds `i32::MAX`.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `args` - [`All2allvArgs`]  object encapsulating send and receive counts.
/// * `comm` - Communicator
pub fn all2allv_via_scatter<T, S>(
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
    let total_send = uargs.snd_cts.iter().sum::<usize>();
    let total_rcv = uargs.rcv_cts.iter().sum::<usize>();
    let local_max = total_send.max(total_rcv);
    let g_max = allreduce(&local_max, comm, SystemOperation::max());
    //  Handle large size
    if g_max > i32::MAX as usize {
        todo!("Handle Big");
    } else {
        all2allv_via_scatter_(s_in, s_out, args, comm)?
    }
    Ok(())
}

/// All-to-allv-via-scatter from raw send/recv count slices.
///
/// # Description
/// Convenience wrapper around [`all2allv_via_scatter`] that builds
/// the [`All2allvArgs`] from the supplied count vectors.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `s_out` - per-rank mutable output slice for all2allv.
/// * `send_counts` - number of elements to send from 
///    this process to each of the other processes.
/// * `recv_counts` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn all2allv_via_scatter_slice<T>(
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
    all2allv_via_scatter(s_in, s_out, &params, comm)
}

/// All-to-allv-via-scatter returning a freshly allocated `Vec<T>`.
///
/// # Description
/// Convenience wrapper around [`all2allv_via_scatter_slice`] that
/// allocates the receive buffer of length `sum(recv_counts)`.
///
/// # Arguments
/// * `s_in` - per-rank input slice for all2allv.
/// * `send_counts` - number of elements to send from 
///    this process to each of the other processes.
/// * `recv_counts` - number of elements to recieve from each of the other 
///    processes.
/// * `comm` - Communicator
pub fn all2allv_via_scatter_vec<T>(
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
    all2allv_via_scatter_slice(
        s_in,
        &mut rcv_vec,
        send_counts,
        recv_counts,
        comm,
    )?;
    Ok(rcv_vec)
}
