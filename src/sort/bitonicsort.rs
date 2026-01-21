use anyhow::{Ok, Result, bail};
use mpi::{
    point_to_point as p2p,
    traits::{Communicator, Equivalence},
};
use num::pow;
use std::{cmp::Ordering, fmt::Debug, marker::PhantomData, ops::Not};

use super::Error;
use crate::reduction::all_same;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Direction {
    Ascend,
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
