//! MPI broadcast functions
//! Broadcast a value or a vector.

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

/// Represents possible errors during broadcast.
#[derive(Error, Debug)]
pub enum Error {
    // Output Slice is less than expected
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    // Input Slice is doesn't match expected length
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
    // General Input Error
    #[error("Input Error:: {0}")]
    InputError(String),
}

/// Broadcast one element via reference.
///
/// # Description
/// Input is mutable reference, input at the root process and output everywhere.  
///
/// # Arguments
/// * `s_inout` - mutable reference input at the root, output everywhere.
/// * `root` - root process to broadcast from
/// * `comm` - Communicator
///
/// # Returns
/// None
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let mut bvalue: usize = if rank == 0 {
///    12
/// else {
///    0
/// };
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

/// Broadcast one element.
///
/// # Description
/// Broadcast one element, input is Some at the root, None everywhere.
///
/// # Arguments
/// * `s_in` - Some(T) at the root, None everywhere.
/// * `root` - root process to broadcast from
/// * `comm` - Communicator
///
/// # Returns
/// broadcasted T or Error with Result<T> 
///
/// # Errors
/// Retuns InputError if root pocess has None input.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let bvalue: Option<usize> = if rank == 0 {
///    Some(12)
/// else {
///    None
/// };
/// let result = bcast_one(bvalue, 0, &c.comm)?;
/// assert_eq!(result, 12);
/// \```
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

/// Broadcast a slice from the root process to all the processes.
///
/// # Description
/// Broadcast a slice from the root process to all the processes. Slice serves
///  as both the input and output.
///
/// # Arguments
/// * `s_inout` - Slice of T, serves as input at root process and output everywhere
/// * `root` - root process to broadcast from
/// * `comm` - Communicator
///
/// # Returns
/// Error Result.
///
/// # Errors
/// Retuns InputError if the all the processes have different inout length.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let result = if rank == 0 {
///    let mut data = vec![1, 2, 3];
///    bcast(&mut data, 0, &c.comm)
///    data
/// } else {
///    let mut data = vec![0, 0, 0];
///    bcast(&mut data, 0, &c.comm)
///    data
/// };
/// assert_eq!(result, Some(vec![1, 2, 3]));
/// \```
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
            "bcast_one input size should be all same.".to_string()
        ))
    }
    // TODO:: handle large sizes
    let root_process = comm.process_at_rank(root);
    root_process.broadcast_into(s_inout);
    Ok(())
}

/// Broadcast a vector from the root process to all the processes.
///
/// # Description
/// Input is an option, the root process shouldn't be None.
///
/// # Arguments
/// * `s_in` - Optional Slice of T, Can not be None at root process 
/// * `root` - root process to broadcast from
/// * `comm` - Communicator
///
/// # Returns
/// Broadcasted vector.
///
/// # Errors
/// Retuns InputError if the root process has None input.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let result = if rank == 0 {
///    let data = vec![1, 2, 3];
///    bcast_vec(Some(&data), 0, &c.comm)
/// } else {
///    bcast_vec(None, 0, &c.comm)
/// };
/// assert_eq!(result, Some(vec![1, 2, 3]));
/// \```
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
            "bcast_one input @ root is None.".to_string()
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
