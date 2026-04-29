//! Parallel sample sort.
//!
//! Implements a classical distributed sample-sort:
//!
//! 1. Locally sort each rank's slice.
//! 2. Pick `s = p - 1` evenly spaced samples per rank
//!    ([`sample_block_decomp`] when the input is already
//!    block-balanced, [`sample_arbit_decomp`] otherwise).
//! 3. Sort the gathered samples with [`bitonic_sort`] and pick the
//!    last sample of every rank (except the last) as the global
//!    splitters; allgather them to obtain `p - 1` splitters
//!    everywhere.
//! 4. Locally bucket each rank's data into `p` buckets using the
//!    splitters ([`split`] for the unstable variant or
//!    [`stable_split`] for the stable variant, the latter routes
//!    runs of equal splitters to a deterministic destination so
//!    that order is preserved).
//! 5. Exchange buckets via `MPI_Alltoallv`
//!    ([`crate::collective::all2allv_vec`]).
//! 6. Locally re-sort the received elements (or multi-way merge).
//! 7. Re-balance into the original distribution using
//!    [`crate::distribution::stable_distribute`] or
//!    [`crate::distribution::arbit_distribute`].
//!
//! Public entry point is [`samplesort`].

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
use mpi::{topology::Communicator, traits::Equivalence};
use std::cmp::Ordering;

use super::Error;
use crate::{
    collective::{all2all_vec, all2allv_vec, allgatherv_vec},
    distribution::{arbit_distribute, distribute_vec, stable_distribute},
    partition::{Dist, ModuloDist},
    reduction::{all_of, allreduce_sum},
    sort::bitonicsort::bitonic_sort,
    util::equal_range_by,
};

/// Pick `p - 1` global splitters when the input is arbitrarily
/// distributed.
///
/// # Description
/// Each rank picks samples at positions proportional to its share
/// of the global size (so larger ranks contribute more samples).
/// The total number of sampled elements is `s * p`. Samples are
/// then re-balanced across processes via
/// [`crate::distribution::distribute_vec`] onto a [`ModuloDist`],
/// truncated to exactly `s` samples per rank, sorted across ranks
/// with [`bitonic_sort`], and the last sample of each rank (except
/// the last one) is allgathered to form the global splitters.
///
/// # Errors
/// * [`super::Error::SampleSizeError`] when the global size is
///   smaller than `s * p`.
/// * [`super::Error::SplitterSizeError`] when the redistributed
///   per-rank splitter count drops below `s`.
pub fn sample_arbit_decomp<T, F>(
    t_in: &[T],
    compare: F,
    s: usize,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let local_size: usize = t_in.len();
    let total_size: usize = allreduce_sum(&local_size, comm);
    let p = comm.size();
    let total_s = s * p as usize;

    // pick s*p samples, i.e. max(ceil((local_size/n)*s*p), 1) locally
    let local_s = if local_size == 0 {
        0
    } else {
        //1usize.max(((local_size * total_s) + total_size - 1) / total_size)
        1usize.max((local_size * total_s).div_ceil(total_size))
    };

    if !all_of(allreduce_sum(&local_size, comm) >= total_s, comm) {
        bail!(Error::SampleSizeError)
    }

    //  pick local samples if (local_s > 0)
    let local_splitters = if local_s > 0 {
        let mut l_splitters: Vec<T> = vec![T::default(); local_s];
        let mut pos: usize = 0;
        for (i, lx) in l_splitters.iter_mut().enumerate() {
            // modulo-based bucket size
            let bucket_size: usize = local_size / (local_s + 1)
                + (i < (local_size % (local_s + 1))) as usize;
            // pick last element of each bucket
            pos += bucket_size - 1;
            *lx = t_in[pos].clone();
            pos += 1;
        }
        l_splitters
    } else {
        vec![]
    };

    //distribute elements equally
    // TODO:: distribute_inplace instaed ?
    let s_part = ModuloDist::new(
        allreduce_sum(&local_splitters.len(), comm),
        p,
        comm.rank(),
    );
    let mut local_splitters = distribute_vec(&local_splitters, &s_part, comm)?;

    // Should have atleast s splitters
    if !all_of(local_splitters.len() >= s, comm) {
        bail!(Error::SplitterSizeError(
            "Number of splitters less than expected.".to_string()
        ));
    }

    // discard extra splitters, to make it even
    if local_splitters.len() != s {
        local_splitters.resize(s, T::default());
    }

    //  sort splitters using parallel bitonic sort
    bitonic_sort(&mut local_splitters, compare, comm)?;

    // select the last element on each process but the last
    let my_splitter: T = local_splitters
        .last()
        .ok_or(Error::MissingLastError)?
        .clone();

    // allgather splitters (from all but the last processor)
    let mut recv_sizes: Vec<i32> = vec![1; comm.size() as usize];
    recv_sizes[comm.size() as usize - 1] = 0;
    let sv = if comm.rank() != comm.size() - 1 {
        vec![my_splitter]
    } else {
        vec![]
    };
    allgatherv_vec(&sv, &recv_sizes, comm)
}

/// Pick `p - 1` global splitters when the input is block-decomposed.
///
/// # Description
/// Faster path used when every rank has roughly the same number of
/// elements (`local_size > 0` everywhere). Each rank picks `s`
/// equally-spaced samples from its already-sorted local slice
/// (one sample per "bucket" of size `local_size / (s + 1)`), the
/// per-rank samples are sorted across ranks with [`bitonic_sort`],
/// and the last sample of each rank (except the last) is
/// allgathered to form the global splitters.
///
/// # Errors
/// [`super::Error::SampleSizeError`] when any rank has an empty
/// local slice.
pub fn sample_block_decomp<T, F>(
    t_in: &mut [T],
    compare: F,
    s: usize,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let local_size: usize = t_in.len();
    if !all_of(local_size > 0, comm) {
        bail!(Error::SampleSizeError)
    }

    // samples
    //  - pick `s` samples equally spaced such that `s` samples define `s+1`
    //    subsequences in the sorted order
    let bucket_f: usize = local_size / (s + 1);
    let mut local_splitters: Vec<T> = vec![T::default(); s];
    let mut pos: usize = 0;
    for (i, lx) in local_splitters.iter_mut().enumerate() {
        let bucket_size = bucket_f + (i < (local_size % (s + 1))) as usize;
        pos += bucket_size - 1;
        *lx = t_in[pos].clone();
        pos += 1;
    }

    // sort splitters using parallel bitonic sort
    bitonic_sort(&mut local_splitters[..], compare, comm)?;

    // select the last element on each process but the last
    let my_splitter: T = local_splitters
        .last()
        .ok_or(Error::MissingLastError)?
        .clone();

    // allgather splitters (from all but the last processor)
    let mut recv_sizes: Vec<i32> = vec![1; comm.size() as usize];
    recv_sizes[comm.size() as usize - 1] = 0;
    let sv = if comm.rank() != comm.size() - 1 {
        vec![my_splitter]
    } else {
        vec![]
    };
    let result_splitters = allgatherv_vec(&sv, &recv_sizes[..], comm)?;
    Ok(result_splitters)
}

/// Compute per-bucket send counts for the unstable variant of
/// sample sort.
///
/// # Description
/// Walks `tsl` in sorted order using [`equal_range_by`] to locate
/// each splitter and assigns elements to processes:
///
/// * elements strictly less than splitter `i` are routed to rank
///   `i`,
/// * runs of elements equal to one or more consecutive splitters
///   are *split fairly* across the relevant ranks based on their
///   target [`ModuloDist`] sizes (so that the resulting
///   distribution is approximately balanced).
///
/// The function returns the per-rank send counts; the caller is
/// expected to perform the all-to-allv exchange. An invariant
/// check verifies that `sum(send_counts) == local_size`.
///
/// # Errors
/// * [`super::Error::SplitterSizeError`] if `splitters.len() != p - 1`.
/// * [`super::Error::SortInvariantError`] if the invariant on
///   `send_counts` is violated.
fn split<T, F>(
    tsl: &mut [T],
    splitters: &[T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<Vec<usize>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if !all_of(splitters.len() == comm.size() as usize - 1, comm) {
        bail!(Error::SplitterSizeError("Size not p - 1.".to_string()));
    }
    // Locally find splitter positions in data
    //  (if an identical splitter appears at least three times (or more),
    //  then split the intermediary buckets evenly) => send_counts

    let local_size = tsl.len();
    let local_part = ModuloDist::new(local_size, comm.size(), comm.rank());
    let mut send_counts: Vec<usize> = vec![0; comm.size() as usize];
    let mut pos: usize = 0;
    let mut i: usize = 0;

    while i < splitters.len() {
        // get the number of splitters which are equal starting from `i`
        let mut split_by: usize = 1;
        while i + split_by < splitters.len()
            && !compare(&splitters[i], &splitters[i + split_by]).is_lt()
        {
            split_by += 1;
        }
        // get the range of equal elements
        let eqr = equal_range_by(tsl, pos, &splitters[i], &compare);
        // assign smaller elements to processor left of splitter (= `i`)
        send_counts[i] += eqr.first - pos;
        pos = eqr.first;
        // split equal elements fairly across processors
        let mut eq_size = eqr.second - pos;
        //  - try to split approx equal:
        let eq_size_split = (eq_size + send_counts[i]) / (split_by + 1) + 1;
        for j in 0..split_by {
            let mut out_size: usize = 0;
            let lpart_size = local_part.local_size_at((i + j) as i32);
            if send_counts[i + j] < lpart_size {
                out_size = usize::min(
                    usize::max(lpart_size - send_counts[i + j], eq_size_split),
                    eq_size,
                );
                eq_size -= out_size;
            }
            send_counts[i + j] += out_size;
        }
        // - assign remaining elements to next processor
        send_counts[i + split_by] += eq_size;
        i += split_by;
        pos = eqr.second;
    }
    // send last elements to last processor
    let out_size = tsl.len() - pos;
    send_counts[comm.size() as usize - 1] += out_size;
    // variant check
    if !all_of(send_counts.iter().sum::<usize>() == local_size, comm) {
        bail!(Error::SortInvariantError(
            "send_counts.iter().sum() === local_size".to_string()
        ))
    }
    Ok(send_counts)
}

/// Compute per-bucket send counts for the stable variant of sample
/// sort.
///
/// # Description
/// Stable counterpart of [`split`]. Elements strictly less than a
/// splitter are routed deterministically to the rank immediately
/// to its left; runs of equal splitters are not split arbitrarily
/// but instead routed to a single deterministic destination (chosen
/// from the rank's position relative to the run length). This
/// guarantees that elements that compare equal preserve their
/// relative order across the global sort.
///
/// # Errors
/// * [`super::Error::SplitterSizeError`] if `splitters.len() != p - 1`.
/// * [`super::Error::SortInvariantError`] if the invariant on
///   `send_counts` is violated.
fn stable_split<T, F>(
    tsl: &mut [T],
    splitters: &[T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<Vec<usize>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if !all_of(splitters.len() == comm.size() as usize - 1, comm) {
        bail!(Error::SplitterSizeError("Splitter not p - 1".to_string()));
    }
    // Locally find splitter positions in data
    //  (if an identical splitter appears at least three times (or more),
    //   then split the intermediary buckets evenly) => send_counts
    let mut send_counts: Vec<usize> = vec![0; comm.size() as usize];
    let mut i: usize = 0;
    let mut pos: usize = 0;

    while i < splitters.len() {
        // get the number of splitters which are equal starting from `i`
        let mut split_by: usize = 1;
        while i + split_by < splitters.len()
            && !compare(&splitters[i], &splitters[i + split_by]).is_lt()
        {
            split_by += 1;
        }
        // get the range of equal elements
        let eqr = equal_range_by(tsl, pos, &splitters[i], &compare);
        // assign smaller elements to processor left of splitter (= `i`)
        send_counts[i] += eqr.first - pos;
        pos = eqr.first;
        // split equal elements fairly across processors
        let eq_size = eqr.second - pos;
        // send equal elements to processor based on my own rank compared to
        // how many equal splitters there are
        if split_by == 1 {
            // Case 1) if there is only one splitter,
            //         assign equal elements to next processor (no splitting)
            send_counts[i + 1] += eq_size;
        } else {
            // Case 2) if there is >= 2 equal splitters:
            //         split processors into `split_by` regions
            let mut targetp =
                (comm.rank() as usize * split_by) / comm.size() as usize;
            if targetp >= split_by {
                targetp = split_by - 1
            };
            send_counts[i + 1 + targetp] += eq_size;
        }
        i += split_by;
        pos = eqr.second;
    }

    // send last elements to last processor
    let out_size = tsl.len() - pos;
    send_counts[comm.size() as usize - 1] += out_size;
    // variant check
    if !all_of(send_counts.iter().sum::<usize>() == tsl.len(), comm) {
        bail!(Error::SortInvariantError(
            "send_counts.iter().sum() === local_size".to_string()
        ))
    }
    Ok(send_counts)
}

/// Distributed parallel sample sort.
///
/// # Description
/// Sorts `tsl` across the communicator. The high-level steps are:
///
/// 1. Local sort (stable or unstable depending on `stable`).
/// 2. Sample `s = p - 1` splitters per rank
///    ([`sample_block_decomp`] or [`sample_arbit_decomp`] depending
///    on whether the input is already block-decomposed).
/// 3. Compute per-rank send counts ([`split`] or [`stable_split`]).
/// 4. Exchange data via `MPI_Alltoallv` and re-sort the received
///    bucket locally.
/// 5. Re-balance into the original distribution
///    ([`crate::distribution::stable_distribute`] for block input,
///    [`crate::distribution::arbit_distribute`] otherwise).
///
/// # Arguments
/// * `tsl` - per-rank slice to sort in place.
/// * `compare` - ordering function.
/// * `stable` - whether to preserve the relative order of equal
///   elements.
/// * `comm` - Communicator
///
/// # Errors
/// Propagates errors from [`super::Error`] (sample size, splitter
/// size, sort invariants) and from the underlying collectives /
/// distributors.
pub fn samplesort<T, F>(
    tsl: &mut [T],
    compare: F,
    stable: bool,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    // sample sort main steps
    // 1. local sort
    // 2. pick `s` samples regularly spaced on each processor
    // 3. bitonic sort samples
    // 4. allgather the last sample of each process -> splitters
    // 5. locally find splitter positions in data
    //    (if an identical splitter appears twice, then split evenly)
    //    => send_counts
    // 6. distribute send_counts with all2all to get recv_counts
    // 7. allocate enough space (may be more than previously allocated) for receiving
    // 8. all2allv
    // 9. local reordering (multiway-merge or again std::sort)
    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by sending elements to neighbors

    // p, rank, no, of samples (p - 1)
    let (p, rank, s) = (comm.size(), comm.rank(), comm.size() as usize - 1);

    // TODO:: timer start;
    if stable {
        tsl.sort_by(&compare);
    } else {
        tsl.sort_unstable_by(&compare);
    }

    // sequential case: we're done
    if p == 1 {
        return Ok(());
    }

    // local size & global sizes
    let local_size: usize = tsl.len();
    let global_size: usize = allreduce_sum(&local_size, comm);

    // check if we have a perfect block decomposition
    let mypart: ModuloDist = ModuloDist::new(global_size, p, rank);
    let _is_block_decomp: bool = all_of(local_size == mypart.local_size(), comm);

    // get splitters, using the method depending on whether the input consists
    // of arbitrary decompositions or not
    let local_splitters: Vec<T> = if _is_block_decomp {
        sample_block_decomp(tsl, &compare, s, comm)?
    } else {
        sample_arbit_decomp(tsl, &compare, s, comm)?
    };

    // 5. locally find splitter positions in data
    //    (if an identical splitter appears at least three times (or more),
    //    then split the intermediary buckets evenly) => send_counts
    let send_counts = if stable {
        stable_split(tsl, &local_splitters, &compare, comm)?
    } else {
        split(tsl, &local_splitters, &compare, comm)?
    };

    // MXX_ASSERT(!_AssumeBlockDecomp || (local_size <= (size_t)p || recv_n <= 2* local_size));
    let recv_counts = all2all_vec(&send_counts, comm)?;
    let recv_n: usize = recv_counts.iter().sum();
    if !all_of(
        !_is_block_decomp
            || (local_size <= p as usize)
            || recv_n <= 2 * local_size,
        comm,
    ) {
        bail!(Error::SortInvariantError(
            "!_is_block_decomp || (local_size <= p as usize) || recv_n <= 2 * local_size".to_string()
        ))
    }
    // TODO: use collective with iterators [begin,end) instead of pointers!
    let mut recv_elts = all2allv_vec(tsl, &send_counts, &recv_counts, comm)?;

    // 9. local reordering
    // TODO::: multi-way merge instead of sort
    if stable {
        recv_elts.sort_by(&compare);
    } else {
        recv_elts.sort_unstable_by(&compare);
    }

    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors
    //    and save elements into the original iterator positions
    if _is_block_decomp {
        stable_distribute(&recv_elts, tsl, &mypart, comm)?;
    } else {
        arbit_distribute(&recv_elts, tsl, local_size, comm)?;
    }
    Ok(())
}
