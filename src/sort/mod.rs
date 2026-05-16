//! Parallel distributed sorting primitives.
//!
//! This module exposes two parallel sorting algorithms:
//!
//! * [`bitonic_sort`] - parallel bitonic sort (in [`bitonicsort`]).
//!   Requires every process to hold the same number of elements; the
//!   total work and communication match the classical bitonic-sort
//!   network.
//! * [`samplesort::samplesort`] - parallel sample sort (in [`mod@samplesort`]).
//!   Picks `p - 1` global splitters by sampling every rank, partitions the
//!   data into `p` buckets and exchanges them with a single
//!   `MPI_Alltoallv`.
//!
//! Convenience wrappers ([`sort`], [`sort_by`], [`stable_sort`],
//! [`stable_sort_by`]) sit on top of [`mod@samplesort`] and select the
//! stable / unstable mode of the underlying local sort.
//!
//! Distributed [`is_sorted`] / [`is_sorted_by`] predicates check that
//! a slice is locally sorted on every rank *and* that the boundary
//! between consecutive ranks respects the comparator (using
//! [`crate::shift::right_shift`] to inspect the previous rank's
//! last element).

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

mod bitonicsort;
mod samplesort;

use super::reduction::all_of;
use super::shift::right_shift;
use anyhow::Result;
use mpi::traits::{Communicator, Equivalence};
use std::cmp::Ordering;
use thiserror::Error;

/// Errors produced by the sorting and is-sorted helpers.
#[derive(Error, Debug)]
pub enum Error {
    /// [`bitonic_sort`] requires every process to hold the same
    /// number of elements; this error is raised when that invariant
    /// is violated across the communicator.
    #[error("Bitonic Sort requires the same number of elements on each process.")]
    BitonicNotEqualError,
    /// A helper expected the local slice to have a last element but
    /// it was empty (e.g. when picking the rank's local splitter).
    #[error("Missing last value in the slice")]
    MissingLastError,
    /// A helper expected the local slice to have a first element but
    /// it was empty.
    #[error("Missing first value in the slice")]
    MissingFirstError,
    /// The number of splitters did not match `comm.size() - 1` (or the
    /// expected count from sampling).
    #[error("Splitter Size Error : {0} ")]
    SplitterSizeError(String),
    /// The total input size across all ranks is not large enough to
    /// pick the requested number of samples in [`samplesort::samplesort`].
    #[error("Insufficient Sample Size")]
    SampleSizeError,
    /// An internal invariant of the sort algorithm did not hold;
    /// usually indicates a bug or a malformed input.
    #[error("Invariant not satisfied: {0}")]
    SortInvariantError(String),
}

/// Distributed `is_sorted_by` predicate.
///
/// # Description
/// Returns `true` only when `s_slice` is locally sorted with respect
/// to `compare` on every rank *and* the rank boundaries agree:
/// every rank `r > 0` must have `compare(prev, s_slice.first())`
/// hold, where `prev` is the last element of rank `r - 1` (obtained
/// via [`right_shift`]).
///
/// # Arguments
/// * `s_slice` - per-rank slice to verity if it is already sorted.
/// * `compare` - ordering function.
/// * `comm` - Communicator
///
/// # Errors
/// * [`Error::MissingLastError`] / [`Error::MissingFirstError`] if a
///   rank's slice is empty and the multi-rank check is needed.
pub fn is_sorted_by<T, F>(
    s_slice: &[T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<bool>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> bool,
{
    let bsorted = s_slice.is_sorted_by(&compare);
    if comm.size() == 1 {
        return Ok(bsorted);
    }
    let lval = s_slice.last().ok_or(Error::MissingLastError)?;
    let fval = s_slice.first().ok_or(Error::MissingFirstError)?;
    let prev = right_shift(lval, comm);
    Ok(all_of(
        if comm.rank() > 0 {
            bsorted && prev.is_some_and(|prev| compare(&prev, fval))
        } else {
            bsorted
        },
        comm,
    ))
}

/// Distributed `is_sorted` using the natural [`Ord`] of `T`.
///
/// # Description
/// Convenience wrapper around [`is_sorted_by`] that uses
/// `T::le` as the comparator (i.e. checks ascending order).
///
/// # Arguments
/// * `s_slice` - per-rank slice to verity if it is already sorted.
/// * `comm` - Communicator
pub fn is_sorted<T>(s_slice: &[T], comm: &dyn Communicator) -> Result<bool>
where
    T: Equivalence + Default + Clone + Ord,
{
    is_sorted_by(s_slice, T::le, comm)
}

pub use bitonicsort::bitonic_sort;
pub use samplesort::samplesort;

/// Distributed sort with a custom comparator (unstable).
///
/// # Description
/// Sorts `tsl` across the communicator using [`samplesort::samplesort`] with
/// the `stable` flag set to `false`. Local sorting falls back to
/// [`slice::sort_unstable_by`].
///
/// # Arguments
/// * `tsl` - per-rank slice to sort in place.
/// * `compare` - ordering function.
/// * `comm` - Communicator
pub fn sort_by<T, F>(
    tsl: &mut [T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    samplesort(tsl, compare, false, comm)
}

/// Distributed sort using the natural [`Ord`] of `T` (unstable).
///
/// # Description
/// Convenience wrapper around [`sort_by`] using `T::cmp`.
///
/// # Arguments
/// * `tsl` - per-rank slice to sort in place.
/// * `comm` - Communicator
pub fn sort<T>(tsl: &mut [T], comm: &dyn Communicator) -> Result<()>
where
    T: Equivalence + Default + Clone + Ord,
{
    samplesort(tsl, T::cmp, false, comm)
}

/// Distributed stable sort with a custom comparator.
///
/// # Description
/// Sorts `tsl` across the communicator using [`samplesort::samplesort`] with
/// the `stable` flag set to `true`. Local sorting uses
/// [`slice::sort_by`] and equal splitters are routed to a
/// deterministic destination (see [`mod@samplesort`]'s `stable_split`).
///
/// # Arguments
/// * `tsl` - per-rank slice to stable sort in place.
/// * `compare` - ordering function.
/// * `comm` - Communicator
pub fn stable_sort_by<T, F>(
    tsl: &mut [T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    samplesort(tsl, compare, true, comm)
}

/// Distributed stable sort using the natural [`Ord`] of `T`.
///
/// # Description
/// Convenience wrapper around [`stable_sort_by`] using `T::cmp`.
///
/// # Arguments
/// * `tsl` - per-rank slice to stable sort in place.
/// * `comm` - Communicator
pub fn stable_sort<T>(tsl: &mut [T], comm: &dyn Communicator) -> Result<()>
where
    T: Equivalence + Default + Clone + Ord,
{
    samplesort(tsl, T::cmp, true, comm)
}
