//! This crate is a Rust port of patflick's [mxx] C++ template library for MPI.
//! Similar to [mxx], the goal is to provide:
//! 1. Simplified, efficient, and type-safe wrappers to common MPI operations
//!    along with input validation and error.
//! 2. Collection of high-performance standard algorithms for parallel
//!    distributed memory.
//!
//! [mxx]: https://github.com/patflick/mxx

pub mod bcast;
pub mod big_collective;
pub mod collective;
pub mod comm;
pub mod distribution;
pub mod log;
pub mod partition;
pub mod reduction;
pub mod shift;
pub mod sort;
pub mod timer;
pub mod util;

use mpi::traits::Equivalence;
use num::{FromPrimitive, Integer, ToPrimitive};
use std::ops::AddAssign;

pub trait MCount:
    Integer + Default + Clone + AddAssign + ToPrimitive + FromPrimitive + Equivalence
{
}
impl<
    T: Integer
        + Default
        + Clone
        + AddAssign
        + ToPrimitive
        + FromPrimitive
        + Equivalence,
> MCount for T
{
}

#[derive(Debug)]
/// All2allv arguments which includes send counts, send displacements,
/// recieve counts and recieve displacements.
pub struct All2allvArgs<T> {
    pub rcv_cts: Vec<T>,
    pub rcv_disp: Vec<T>,
    pub snd_cts: Vec<T>,
    pub snd_disp: Vec<T>,
}

impl<T> All2allvArgs<T>
where
    T: 'static + MCount,
{
    // Creates an empty All2allvArgs object with all members
    pub fn new(p: usize) -> Self {
        All2allvArgs {
            rcv_cts: vec![T::default(); p],
            rcv_disp: vec![T::default(); p],
            snd_cts: vec![T::default(); p],
            snd_disp: vec![T::default(); p],
        }
    }

    // Creates an object with provided counts, and displacements computed with
    // exclusive prefix sum based on the counts
    pub fn from_counts<S: ToPrimitive>(
        send_counts: &[S],
        recv_counts: &[S],
    ) -> Self {
        use crate::util::exc_prefix_sum_iter;
        let snd_cts: Vec<T> = send_counts
            .iter()
            .map(|x| T::from_usize(x.to_usize().unwrap()).unwrap())
            .collect();
        let rcv_cts: Vec<T> = recv_counts
            .iter()
            .map(|x| T::from_usize(x.to_usize().unwrap()).unwrap())
            .collect();
        let snd_disp = exc_prefix_sum_iter(snd_cts.iter(), T::one()).collect();
        let rcv_disp = exc_prefix_sum_iter(rcv_cts.iter(), T::one()).collect();
        All2allvArgs::<T> {
            snd_cts,
            snd_disp,
            rcv_cts,
            rcv_disp,
        }
    }

    // Creates an All2allvArgs<i32> object from the existing object
    pub fn to_i32(&self) -> All2allvArgs<i32> {
        All2allvArgs::<i32> {
            rcv_cts: self.rcv_cts.iter().map(|x| x.to_i32().unwrap()).collect(),
            rcv_disp: self.rcv_disp.iter().map(|x| x.to_i32().unwrap()).collect(),
            snd_cts: self.snd_cts.iter().map(|x| x.to_i32().unwrap()).collect(),
            snd_disp: self.snd_disp.iter().map(|x| x.to_i32().unwrap()).collect(),
        }
    }

    // Creates an All2allvArgs<usize> object from the existing object
    pub fn to_usize(&self) -> All2allvArgs<usize> {
        All2allvArgs::<usize> {
            rcv_cts: self.rcv_cts.iter().map(|x| x.to_usize().unwrap()).collect(),
            rcv_disp: self
                .rcv_disp
                .iter()
                .map(|x| x.to_usize().unwrap())
                .collect(),
            snd_cts: self.snd_cts.iter().map(|x| x.to_usize().unwrap()).collect(),
            snd_disp: self
                .snd_disp
                .iter()
                .map(|x| x.to_usize().unwrap())
                .collect(),
        }
    }
}

pub mod traits {
    pub use sope_derive::GEquivalence;
}
