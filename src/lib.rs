//! This crate is a Rust port of patflick's [mxx] C++ template library for MPI.
//! Similart to [mxx], the goal is to provide:
//! 1. Simplified, efficient, and type-safe wrappers to common MPI operations
//!    along with input validation and error.
//! 2. Collection of high-performance standard algorithms for parallel
//!    distributed memory.
//!
//! [mxx]: https://github.com/patflick/mxx

pub mod collective;

/// Interface to Comm objects
pub mod bcast;
pub mod comm;
pub mod distribution;
pub mod log;
pub mod partition;
pub mod reduction;
pub mod shift;
pub mod sort;
pub mod util;
