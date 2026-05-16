//! Typed MPI broadcast helpers.
//!
//! Wrappers around `MPI_Bcast` that accept Rust scalars, slices and
//! `Vec`s instead of raw datatype handles. There are three flavours,
//! covering the most common patterns:
//!
//! * [`bcast_one_ref`] - lowest-level helper. Both root and
//!   non-root ranks pass a mutable reference; the root's value is
//!   broadcast in place to every rank.
//! * [`bcast_one`] / [`bcast_vec`] - "optional" inputs: the root
//!   passes `Some(value)` and every other rank passes `None`. The
//!   helper returns the broadcast value on every rank. For
//!   [`bcast_vec`] the slice length is broadcast first (using
//!   [`bcast_one`]) so non-root ranks can allocate a buffer of the
//!   correct size.
//! * [`bcast`] - in-place broadcast over a caller-supplied slice
//!   that must already be the same length on every rank.
//!
//! All helpers validate their inputs collectively across the
//! communicator (using [`crate::reduction::any_of`] /
//! [`crate::reduction::all_same`]) before calling into `rsmpi`, and
//! return `anyhow::Result<...>` with a typed [`enum@Error`] on failure.

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

use crate::reduction::{all_same, any_of};
use anyhow::{Ok, Result, bail};
use mpi::traits::{Communicator, Equivalence, Root};
use thiserror::Error;

/// Errors raised by the broadcast helpers.
#[derive(Error, Debug)]
pub enum Error {
    /// The supplied output slice is shorter than the broadcast
    /// length.
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    /// The supplied input slice does not match the expected length
    /// (e.g. ranks disagree on the slice length passed to
    /// [`bcast`]).
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
    /// Generic input precondition failure (typically: the root
    /// rank's input is `None` for [`bcast_one`] / [`bcast_vec`]).
    #[error("Input Error:: {0}")]
    InputError(String),
}

/// Broadcast one element through a mutable reference.
///
/// # Description
/// On entry, `s_inout` holds the value to broadcast on the root
/// rank and is undefined elsewhere. On exit, every rank sees the
/// same value as the root. This is the lowest-level wrapper and
/// performs no input validation; prefer [`bcast_one`] for the
/// `Option`-based API.
///
/// # Arguments
/// * `s_inout` - mutable reference; input at the root, output
///   everywhere.
/// * `root` - rank to broadcast from.
/// * `comm` - Communicator
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init();
/// let mut bvalue: usize = if c.rank == 0 { 12 } else { 0 };
/// bcast_one_ref(&mut bvalue, 0, &c.comm);
/// assert_eq!(bvalue, 12);
/// \```
pub fn bcast_one_ref<T>(
    s_inout: &mut T, // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) where
    T: Equivalence,
{
    let root_process = comm.process_at_rank(root);
    root_process.broadcast_into(s_inout);
}

/// Broadcast one element using an `Option`-based API.
///
/// # Description
/// On the root rank, `s_in` must be `Some(value)`; on every other
/// rank it must be `None`. Returns `value` on every rank.
///
/// # Arguments
/// * `s_in` - `Some(T)` at the root, `None` everywhere else.
/// * `root` - rank to broadcast from.
/// * `comm` - Communicator
///
/// # Returns
/// The broadcast `T`, identical on every rank.
///
/// # Errors
/// [`Error::InputError`] when the root rank passes `None`.
///
/// # Examples
/// ```
/// let c = crate::comm::WorldComm::init();
/// let bvalue: Option<usize> = if c.rank == 0 { Some(12) } else { None };
/// let result = bcast_one(bvalue, 0, &c.comm)?;
/// assert_eq!(result, 12);
/// ```
pub fn bcast_one<T>(
    s_in: Option<T>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<T>
where
    T: Equivalence + Default + Clone,
{
    if !any_of(comm.rank() == root && s_in.is_some(), comm) {
        bail!(Error::InputError(
            "bcast_one input @ root is None.".to_string()
        ))
    }
    let mut t_inout: T = if comm.rank() == root {
        s_in.unwrap_or_default()
    } else {
        T::default()
    };
    bcast_one_ref(&mut t_inout, root, comm);
    Ok(t_inout)
}

/// Broadcast a slice in place from the root rank to every rank.
///
/// # Description
/// `s_inout` serves as the input on the root rank and as the
/// output on every rank. All ranks must pass a slice of the same
/// length (this is checked collectively via
/// [`crate::reduction::all_same`]).
///
/// # Arguments
/// * `s_inout` - slice; input at the root rank, output everywhere.
/// * `root` - rank to broadcast from.
/// * `comm` - Communicator
///
/// # Errors
/// [`Error::InputError`] when ranks disagree on the slice length.
///
/// # Examples
/// ```
/// let c = crate::comm::WorldComm::init();
/// let mut data: Vec<i32> = if c.rank == 0 {
///     vec![1, 2, 3]
/// } else {
///     vec![0, 0, 0]
/// };
/// bcast(&mut data, 0, &c.comm)?;
/// assert_eq!(data, vec![1, 2, 3]);
/// ```
pub fn bcast<T>(
    s_inout: &mut [T], // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence,
{
    if !all_same(&s_inout.len(), comm) {
        bail!(Error::InputError(
            "bcast input slice length should be the same on every rank."
                .to_string()
        ))
    }
    // TODO:: handle large sizes
    let root_process = comm.process_at_rank(root);
    root_process.broadcast_into(s_inout);
    Ok(())
}

/// Broadcast a slice from the root rank, returning a `Vec<T>`.
///
/// # Description
/// On the root rank `s_in` must be `Some(slice)`; on every other
/// rank it must be `None`. The slice length is broadcast first
/// (using [`bcast_one`]) so that non-root ranks can allocate a
/// receive buffer of the correct size; the contents are then
/// broadcast with [`bcast`]. The resulting `Vec<T>` is returned on
/// every rank.
///
/// # Arguments
/// * `s_in` - `Some(slice)` at the root rank, `None` everywhere
///   else.
/// * `root` - rank to broadcast from.
/// * `comm` - Communicator
///
/// # Returns
/// A `Vec<T>` containing the broadcast slice, identical on every
/// rank.
///
/// # Errors
/// [`Error::InputError`] when the root rank passes `None`.
///
/// # Examples
/// ```
/// let c = crate::comm::WorldComm::init();
/// let result = if c.rank == 0 {
///     let data = vec![1, 2, 3];
///     bcast_vec(Some(&data), 0, &c.comm)?
/// } else {
///     bcast_vec::<i32>(None, 0, &c.comm)?
/// };
/// assert_eq!(result, vec![1, 2, 3]);
/// ```
pub fn bcast_vec<T>(
    s_in: Option<&[T]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    if !any_of(comm.rank() == root && s_in.is_some(), comm) {
        bail!(Error::InputError(
            "bcast_vec input @ root is None.".to_string()
        ))
    }
    let n = bcast_one(s_in.map(|x| x.len()), root, comm)?;
    let mut v_inout: Vec<T> = if comm.rank() == root {
        s_in.unwrap_or_default().to_vec()
    } else {
        vec![T::default(); n]
    };
    bcast(&mut v_inout, root, comm)?;
    Ok(v_inout)
}
