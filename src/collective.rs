use anyhow::{Ok, Result, bail};
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::{Communicator, CommunicatorCollectives, Equivalence, Root},
};
use num::{FromPrimitive, Integer, ToPrimitive};
use std::{iter::zip, ops::AddAssign};
use thiserror::Error;

use crate::{
    reduction::{all_of, all_same, any_of},
    util::exc_prefix_sum_iter,
};

#[derive(Error, Debug)]
pub enum Error {
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
}

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
}

pub fn scatter_one<T>(
    s_in: Option<&[T]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<T>
where
    T: Equivalence + Default + Clone,
{
    let s_in = s_in.unwrap_or(&[]);
    if !any_of(
        comm.rank() == root && s_in.len() >= comm.size() as usize,
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatter_one input @ root should be >= p.".to_string()
        ));
    }
    let mut rt = T::default();
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        root_process.scatter_into_root(s_in, &mut rt);
    } else {
        root_process.scatter_into(&mut rt);
    }
    Ok(rt)
}

pub fn scatter<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let s_in = s_in.unwrap_or(&[]);
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && s_in.len().is_multiple_of(comm.size() as usize),
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatter input size @ root should be non-zero and a multipe of p."
                .to_string()
        ))
    }
    let mut exp_size = if comm.rank() == root {
        s_in.len() / comm.size() as usize
    } else {
        0
    };

    if !all_same(
        &(if comm.rank() == root {
            exp_size
        } else {
            s_out.len()
        }),
        comm,
    ) {
        let root_process = comm.process_at_rank(root);
        root_process.broadcast_into(&mut exp_size);
        bail!(Error::OutSliceLengthError(exp_size, s_out.len()));
    }

    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        root_process.scatter_into_root(s_in, s_out);
    } else {
        root_process.scatter_into(s_out);
    }
    Ok(())
}

pub fn scatter_vec<T>(
    s_in: Option<&[T]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    // TODO:: handle large sizes
    let s_in = s_in.unwrap_or(&[]);
    let mut exp_size = if comm.rank() == root {
        s_in.len() / comm.size() as usize
    } else {
        0
    };
    comm.process_at_rank(root).broadcast_into(&mut exp_size);
    let mut v_out: Vec<T> = vec![T::default(); exp_size];
    scatter(Some(s_in), &mut v_out, root, comm)?;
    Ok(v_out)
}

pub fn scatterv<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    // TODO:: handle large sizes
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && send_sizes.len() >= comm.size() as usize
            && s_in.len() >= send_sizes.iter().sum::<i32>() as usize,
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatterv input size @ root should be >= sum of send_sizes"
                .to_string()
        ))
    }
    let o_size = scatter_one(Some(send_sizes), root, comm)? as usize;
    if !all_of(
        if o_size == 0 {
            s_out.is_empty()
        } else {
            s_out.len() >= o_size
        },
        comm,
    ) {
        bail!(Error::OutSliceLengthError(o_size, s_out.len()));
    }

    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let displs: Vec<i32> =
            exc_prefix_sum_iter(send_sizes.iter(), 1).collect();
        let partition = Partition::new(s_in, send_sizes, displs);
        root_process.scatter_varcount_into_root(&partition, s_out);
    } else {
        root_process.scatter_varcount_into(s_out);
    }
    Ok(())
}

pub fn scatterv_vec<T>(
    s_in: Option<&[T]>,
    send_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let rcv_size = scatter_one(send_sizes, root, comm)? as usize;
    let mut rcv_vec = vec![T::default(); rcv_size];
    scatterv(s_in, &mut rcv_vec, send_sizes, root, comm)?;
    Ok(rcv_vec)
}

pub fn gather_one<T>(
    s_in: &T,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let mut rcv_vec = vec![T::default(); comm.size() as usize];
        root_process.gather_into_root(s_in, &mut rcv_vec);
        Ok(Some(rcv_vec))
    } else {
        root_process.gather_into(s_in);
        Ok(None)
    }
}

pub fn gather<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let s_out = s_out.unwrap_or(&mut []);
    if !all_same(&(s_in.len()), comm) {
        bail!(Error::InSliceError(
            "gather input sizes should be same across all processors".to_string()
        ))
    }
    let root_process = comm.process_at_rank(root);
    let exp_size = s_in.len() * comm.size() as usize;
    if !any_of(comm.rank() == root && exp_size <= s_out.len(), comm) {
        bail!(Error::OutSliceLengthError(exp_size, s_out.len()));
    }

    if comm.rank() == root {
        root_process.gather_into_root(s_in, s_out);
    } else {
        root_process.gather_into(s_in);
    }
    Ok(())
}

pub fn gather_vec<T>(
    s_in: &[T],
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let mut out_vec = vec![T::default(); s_in.len() * comm.size() as usize];
        gather(s_in, Some(&mut out_vec), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gather(s_in, None, root, comm)?;
        Ok(None)
    }
}

pub fn gatherv<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let s_len = scatter_one(recv_sizes, root, comm)? as usize;
    let i_len = s_in.len();
    if !all_of(
        if s_len == 0 {
            s_in.is_empty()
        } else {
            i_len >= s_len
        },
        comm,
    ) {
        bail!(Error::InSliceError(format!(
            "gather input size should be atleast recv_sizes @ root: R({s_len}) != IN({i_len})."
        )))
    }

    let s_out = s_out.unwrap_or(&mut []);
    let recv_sizes = recv_sizes.unwrap_or(&[]);
    let exp_osize = recv_sizes.iter().sum::<i32>() as usize;
    if !any_of(
        comm.rank() == root && exp_osize > 0 && exp_osize <= s_out.len(),
        comm,
    ) {
        bail!(Error::OutSliceLengthError(exp_osize, s_out.len()));
    }

    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let displs: Vec<i32> =
            exc_prefix_sum_iter(recv_sizes.iter(), 1).collect();
        let mut partition = PartitionMut::new(s_out, recv_sizes, displs);
        root_process.gather_varcount_into_root(s_in, &mut partition);
    } else if !s_in.is_empty() {
        root_process.gather_varcount_into(s_in);
    }
    Ok(())
}

pub fn gatherv_vec<T>(
    s_in: &[T],
    recv_sizes: Option<&[i32]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(&[]);
        let mut out_vec =
            vec![T::default(); recv_sizes.iter().sum::<i32>() as usize];
        gatherv(s_in, Some(&mut out_vec), Some(recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

pub fn gatherv_full_vec<T>(
    s_in: &[T],
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    let ilen: i32 = s_in.len() as i32;
    let recv_sizes = gather_one(&ilen, root, comm)?;
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(vec![]);
        let mut out_vec =
            vec![T::default(); recv_sizes.iter().sum::<i32>() as usize];
        gatherv(s_in, Some(&mut out_vec), Some(&recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

pub fn gather_strings(
    x: String,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<String>>> {
    let lengths: Option<Vec<i32>> = gather_one(&(x.len() as i32), root, comm)?;
    let g_in =
        gatherv_vec(x.as_bytes(), lengths.as_ref().map(|x| &x[..]), root, comm)?;
    if let (Some(sv), Some(lengths)) = (g_in, lengths) {
        let displs: Vec<i32> = exc_prefix_sum_iter(lengths.iter(), 1).collect();
        let svec: Vec<String> = zip(displs.iter(), lengths.iter())
            .map(|(s, l)| {
                let (ts, tl) = (*s as usize, *l as usize);
                String::from_utf8(sv[ts..(ts + tl)].to_vec()).unwrap_or_default()
            })
            .filter(|x| !x.is_empty())
            .collect();
        Ok(Some(svec))
    } else {
        Ok(None)
    }
}

pub fn allgather_one<T>(g_in: &T, comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut g_out = vec![T::default(); comm.size() as usize];
    comm.all_gather_into(g_in, &mut g_out);
    Ok(g_out)
}

pub fn allgather<T>(
    g_in: &[T],
    g_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    if !all_same(&g_in.len(), comm) {
        bail!(Error::InSliceError(
            "allgather input size should be same across all procs.".to_string()
        ));
    }
    let exp_len = g_in.len() * comm.size() as usize;
    if !all_of(g_out.len() == exp_len, comm) {
        bail!(Error::OutSliceLengthError(exp_len, g_out.len()));
    }
    comm.all_gather_into(g_in, g_out);
    Ok(())
}

pub fn allgather_vec<T>(g_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut g_out = vec![T::default(); g_in.len() * comm.size() as usize];
    allgather(g_in, &mut g_out, comm)?;
    Ok(g_out)
}

pub fn allgatherv<T>(
    g_in: &[T],
    g_out: &mut [T],
    recv_sizes: &[i32],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    // TODO:: handle large sizes
    let r_len = recv_sizes[comm.rank() as usize] as usize;
    if !all_of(
        if r_len == 0 {
            g_in.is_empty()
        } else {
            g_in.len() >= r_len
        },
        comm,
    ) {
        bail!(Error::InSliceError(
            "gatherv input size should be at least the total recieve sizes."
                .to_string()
        ))
    }

    let exp_len = recv_sizes.iter().sum::<i32>() as usize;
    if !all_of(g_out.len() >= exp_len, comm) {
        bail!(Error::OutSliceLengthError(exp_len, g_out.len()));
    }

    let displs: Vec<i32> = exc_prefix_sum_iter(recv_sizes.iter(), 1).collect();
    let mut partition = PartitionMut::new(g_out, recv_sizes, displs);
    comm.all_gather_varcount_into(g_in, &mut partition);
    Ok(())
}

pub fn allgatherv_vec<T>(
    g_in: &[T],
    recv_sizes: &[i32],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let out_len = recv_sizes.iter().sum::<i32>() as usize;
    let mut g_out = vec![T::default(); out_len];
    allgatherv(g_in, &mut g_out, recv_sizes, comm)?;
    Ok(g_out)
}

pub fn allgatherv_full_vec<T>(
    s_in: &[T],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let ilen: i32 = s_in.len() as i32;
    let recv_sizes = allgather_one(&ilen, comm)?;
    allgatherv_vec(s_in, &recv_sizes, comm)
}

pub fn all2all<T>(
    a_in: &[T],
    a_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    if !all_of(
        !a_in.is_empty() && a_in.len().is_multiple_of(comm.size() as usize),
        comm,
    ) {
        bail!(Error::InSliceError(
            "all2all input len should be multiple of p.".to_string()
        ));
    }
    if !all_of(a_out.len() == a_in.len(), comm) {
        bail!(Error::OutSliceLengthError(a_in.len(), a_out.len()));
    }
    comm.all_to_all_into(a_in, a_out);
    Ok(())
}

pub fn all2all_vec<T>(a_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut recv_buf: Vec<T> = vec![T::default(); a_in.len()];
    comm.all_to_all_into(a_in, &mut recv_buf);
    Ok(recv_buf)
}

pub fn all2allv<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    // TODO: Handle large size
    let iargs = args.to_i32();
    let send_part = Partition::new(s_in, &iargs.snd_cts[..], &iargs.snd_disp[..]);
    let mut rcv_part =
        PartitionMut::new(s_out, &iargs.rcv_cts[..], &iargs.rcv_disp[..]);
    comm.all_to_all_varcount_into(&send_part, &mut rcv_part);
    Ok(())
}

pub fn all2allv_slice<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    let send_total: usize = send_counts.iter().sum();
    if !all_of(
        if send_total == 0 {
            s_in.is_empty()
        } else {
            s_in.len() >= send_total
        },
        comm,
    ) {
        bail!(Error::InSliceError(
            "all2allv input slice length should be sum of send counts"
                .to_string()
        ));
    }
    let recv_total: usize = recv_counts.iter().sum();
    if !all_of(
        if recv_total == 0 {
            s_out.is_empty()
        } else {
            recv_total <= s_out.len()
        },
        comm,
    ) {
        bail!(Error::OutSliceLengthError(recv_total, s_out.len()));
    }

    let params = All2allvArgs::<usize>::from_counts(send_counts, recv_counts);
    all2allv(s_in, s_out, &params, comm)
}

pub fn all2allv_vec<T>(
    s_in: &[T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let recv_total: usize = recv_counts.iter().sum();
    let mut rcv_vec = vec![T::default(); recv_total];
    all2allv_slice(s_in, &mut rcv_vec, send_counts, recv_counts, comm)?;
    Ok(rcv_vec)
}
