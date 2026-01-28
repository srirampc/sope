use crate::reduction::{all_same, any_of};
use anyhow::{Ok, Result, bail};
use mpi::traits::{Communicator, Equivalence, Root};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
    #[error("Input Error:: {0}")]
    InputError(String),
}

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
