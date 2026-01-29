mod bitonicsort;
mod samplesort;

use super::reduction::all_of;
use super::shift::right_shift;
use anyhow::Result;
use mpi::traits::{Communicator, Equivalence};
use std::cmp::Ordering;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Bitonic Sort requires the same number of elements on each process.")]
    BitonicNotEqualError,
    #[error("Missing last value in the slice")]
    MissingLastError,
    #[error("Missing first value in the slice")]
    MissingFirstError,
    #[error("Splitter Size Error : {0} ")]
    SplitterSizeError(String),
    #[error("Insufficient Sample Size")]
    SampleSizeError,
    #[error("Invariant not satisfied: {0}")]
    SortInvariantError(String),
}

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

pub fn is_sorted<T>(s_slice: &[T], comm: &dyn Communicator) -> Result<bool>
where
    T: Equivalence + Default + Clone + Ord,
{
    is_sorted_by(s_slice, T::le, comm)
}

pub use bitonicsort::bitonic_sort;
pub use samplesort::samplesort;

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

pub fn sort<T>(tsl: &mut [T], comm: &dyn Communicator) -> Result<()>
where
    T: Equivalence + Default + Clone + Ord,
{
    samplesort(tsl, T::cmp, false, comm)
}

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

pub fn stable_sort<T>(tsl: &mut [T], comm: &dyn Communicator) -> Result<()>
where
    T: Equivalence + Default + Clone + Ord,
{
    samplesort(tsl, T::cmp, true, comm)
}
