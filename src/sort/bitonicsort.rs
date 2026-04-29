//! Parallel bitonic sort.
//!
//! Implements a recursive distributed bitonic sort across the
//! communicator. Each rank holds the same number of elements; pairs
//! of ranks repeatedly exchange their full local slices and merge
//! them in either ascending or descending [`Direction`] depending on
//! their relative position. The algorithm follows the classical
//! Batcher's bitonic-sort network, generalised to non-power-of-two
//! process counts (the smaller half is always rounded up to the
//! nearest power of two and the partner-less ranks recurse without
//! exchanging data).
//!
//! Public entry point is [`bitonic_sort`]; the helpers
//! [`bitonic_split`], [`bitonic_merge`] and `bitonic_sort_rec` are
//! internal.

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
    point_to_point as p2p,
    traits::{Communicator, Equivalence},
};
use num::pow;
use std::{cmp::Ordering, fmt::Debug, marker::PhantomData, ops::Not};

use super::Error;
use crate::reduction::all_same;

/// Sort direction used internally by the bitonic algorithm.
///
/// # Description
/// At each recursive step half of the ranks merge in the opposite
/// direction of the other half so that the concatenated result is a
/// bitonic sequence; an outer pass then merges the two halves into a
/// monotonically sorted sequence in the requested direction.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Direction {
    /// Sort in ascending order.
    Ascend,
    /// Sort in descending order.
    Descend,
}

impl Direction {
    pub fn reverse(&self) -> Self {
        match self {
            Direction::Ascend => Direction::Descend,
            Direction::Descend => Direction::Ascend,
        }
    }
}

impl Not for Direction {
    type Output = Self;
    fn not(self) -> Self::Output {
        self.reverse()
    }
}

///
/// Bitonic comparator, a wrapper object that allows for recursively passing
/// the compartor function since each closure creates a new object that 
/// can cause recursion limit.
struct BitonicCompartor<T, F>
where
    T: Default,
    F: Fn(&T, &T) -> Ordering,
{
    phantom: PhantomData<T>,
    compare: F,
}

impl<T, F> BitonicCompartor<T, F>
where
    T: Default,
    F: Fn(&T, &T) -> Ordering,
{
    fn new(compare: F) -> Self {
        Self {
            compare,
            phantom: PhantomData,
        }
    }

    fn cmp(&self, a: &T, b: &T) -> Ordering {
        (self.compare)(a, b)
    }
}

/// One step of the distributed bitonic split.
///
/// # Description
/// Exchange the local slice with the `partner` rank using
/// `MPI_Sendrecv` ([`p2p::send_receive_into`]) and then merge the two
/// equal-length sequences. Depending on the relative ordering of
/// `comm.rank()` and `partner` and the requested `dir`, the merged
/// output keeps either the smaller or the larger half of the
/// combined elements:
///
/// * `(Ascend, partner > rank)` or `(Descend, partner < rank)` keep
///   the smaller half (forward merge).
/// * Otherwise keep the larger half (reverse merge).
///
/// The merge writes into a local buffer first and then copies back
/// into `s_slice`, ensuring the same length on output.
fn bitonic_split<T, F>(
    s_slice: &mut [T],
    b: &BitonicCompartor<T, F>,
    comm: &dyn Communicator,
    partner: i32,
    dir: Direction,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let np = s_slice.len();
    let mut merge_buf = vec![T::default(); np];
    let mut rcv_buf = vec![T::default(); np];
    let partner_process = &comm.process_at_rank(partner);
    p2p::send_receive_into(
        s_slice,
        partner_process,
        &mut rcv_buf,
        partner_process,
    );

    // merge in `dir` direction into merge buffer
    if (dir == Direction::Ascend && partner > comm.rank())
        || (dir == Direction::Descend && partner < comm.rank())
    {
        let mut l_itr = s_slice.iter().peekable();
        let mut r_itr = rcv_buf.iter().peekable();
        for omx in merge_buf.iter_mut() {
            if let (Some(leftv), Some(rightv)) = (l_itr.peek(), r_itr.peek()) {
                if b.cmp(leftv, rightv) == Ordering::Less {
                    *omx = (*leftv).clone();
                    l_itr.next();
                } else {
                    *omx = (*rightv).clone();
                    r_itr.next();
                }
            } else {
                break;
            }
        }
    } else {
        let mut l_itr = s_slice.iter().rev().peekable();
        let mut r_itr = rcv_buf.iter().rev().peekable();
        for omx in merge_buf.iter_mut().rev() {
            if let (Some(leftv), Some(rightv)) = (l_itr.peek(), r_itr.peek()) {
                if b.cmp(leftv, rightv) == Ordering::Less {
                    *omx = (*rightv).clone();
                    r_itr.next();
                } else {
                    *omx = (*leftv).clone();
                    l_itr.next();
                }
            } else {
                break;
            }
        }
    }

    s_slice.clone_from_slice(&merge_buf[..]);
    Ok(())
}

/// Recursively merge a bitonic sequence on ranks `[pbegin, pend)`.
///
/// # Description
/// Given a contiguous range of ranks holding a bitonic sequence,
/// pairs ranks that are `p2/2` apart (where `p2` is the smallest
/// power of two `>= pend - pbegin`) and runs [`bitonic_split`] on
/// each pair, then recurses on the two halves. Ranks without a
/// partner in the upper half still recurse so that the recursion
/// tree stays balanced.
fn bitonic_merge<T, F>(
    s_slice: &mut [T],
    b: &BitonicCompartor<T, F>,
    comm: &dyn Communicator,
    pbegin: i32,
    pend: i32,
    dir: Direction,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let size = pend - pbegin;
    if size <= 1 {
        return Ok(());
    }

    let l2size = f64::ceil(f64::log2(size as f64)) as usize;
    let p2: i32 = pow(2i32, l2size);
    let pmid: i32 = pbegin + p2 / 2;
    if comm.rank() < pmid && (comm.rank() + p2 / 2 < pend) {
        // this processor has a partner in the second half
        let partner_rank: i32 = comm.rank() + p2 / 2;
        bitonic_split(s_slice, b, comm, partner_rank, dir.clone())?;
        bitonic_merge(s_slice, b, comm, pbegin, pmid, dir.clone())?;
    } else if comm.rank() < pmid {
        // this process doesn't have a partner but has to recursively
        // participate in the next merge
        bitonic_merge(s_slice, b, comm, pbegin, pmid, dir)?;
    } else {
        // if (comm.rank() >= pmid) 
        //   partner to the  comm.rank() + p2/2
        let partner_rank: i32 = comm.rank() - p2 / 2;
        bitonic_split(s_slice, b, comm, partner_rank, dir.clone())?;
        bitonic_merge(s_slice, b, comm, pmid, pend, dir.clone())?;
    }
    Ok(())
}

/// Recursive driver of the distributed bitonic sort.
///
/// # Description
/// Sorts the slice across ranks `[pbegin, pbegin + size)` in `dir`
/// order. Splits the range at `pbegin + p2/2` (where `p2` is the
/// next power of two `>= size`):
///
/// * the lower half is sorted in the *opposite* direction (`!dir`),
/// * the upper half is sorted in the requested `dir`,
///
/// producing a bitonic sequence which is then merged across the
/// whole range via [`bitonic_merge`]. Bottoms out when the range
/// contains a single rank.
fn bitonic_sort_rec<T, F>(
    s_slice: &mut [T],
    b: &BitonicCompartor<T, F>,
    comm: &dyn Communicator,
    pbegin: i32,
    size: i32,
    dir: Direction,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    // get next power of two
    let l2size = f64::ceil(f64::log2(size as f64)) as usize;
    let p2: i32 = pow(2i32, l2size);

    // determine where the two sub-recursions are
    let pmid: i32 = pbegin + p2 / 2;
    let mut pend: i32 = pbegin + p2;
    if pend > comm.size() {
        pend = comm.size();
    }
    
    // recursive base-case
    if pend - pbegin <= 1 {
        return Ok(());
    }

    // sort the two subsequences, where the first is always a power of 2
    if comm.rank() < pmid {
        // sort descending
        bitonic_sort_rec(s_slice, b, comm, pbegin, p2 / 2, !dir.clone())?;
    } else {
        // sort ascending
        bitonic_sort_rec(s_slice, b, comm, pmid, size - p2 / 2, dir.clone())?;
    }
    // merge bitonic decreasing sequence
    bitonic_merge(s_slice, b, comm, pbegin, pend, dir.clone())?;

    Ok(())
}

/// Distributed parallel bitonic sort.
///
/// # Description
/// Sorts `s_slice` in ascending `compare` order across the
/// communicator. Every rank must hold the same number of elements
/// (this is checked via [`all_same`]). The local slice is first
/// sorted with the standard library if it is not already sorted,
/// then [`bitonic_sort_rec`] is invoked over the full rank range to
/// run the distributed bitonic-sort network.
///
/// # Arguments
/// * `s_slice` - per-rank slice to sort in place.
/// * `compare` - ordering function (`Ordering::Less` means strictly
///   less).
/// * `comm` - Communicator
///
/// # Errors
/// [`super::Error::BitonicNotEqualError`] when the per-rank slice
/// lengths differ.
pub fn bitonic_sort<T, F>(
    s_slice: &mut [T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let np = s_slice.len();
    // check if all
    if !all_same(&np, comm) {
        bail!(Error::BitonicNotEqualError);
    };

    if !s_slice.is_sorted_by(|a, b| compare(a, b) == Ordering::Less) {
        s_slice.sort_by(&compare);
    }
    let b = BitonicCompartor::<T, F>::new(compare);

    bitonic_sort_rec(s_slice, &b, comm, 0, comm.size(), Direction::Ascend)?;
    Ok(())
}
