use anyhow::{Ok, Result, bail};
use mpi::{
    collective::SystemOperation,
    datatype::{Partition, PartitionMut},
    traits::{Communicator, CommunicatorCollectives, Equivalence, Root},
};
use num::ToPrimitive;
use std::iter::zip;
use thiserror::Error;

use crate::{
    All2allvArgs, MCount,
    reduction::{all_of, all_same, allreduce, any_of},
    util::exc_prefix_sum_iter,
};

#[derive(Error, Debug)]
pub enum Error {
    #[error("Output Slice Length:: Expected {0}, Found {1}")]
    OutSliceLengthError(usize, usize),
    #[error("Input Slice Error:: {0}")]
    InSliceError(String),
}

pub fn validate_all2all<T>(
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
    Ok(())
}

pub fn validate_all2allv<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
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
    Ok(())
}

pub fn validate_scatterv<T, S>(
    s_in: Option<&[T]>,
    s_out: &[T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[S]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<S>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    let s_total = send_sizes
        .iter()
        .map(|x| x.to_usize().unwrap_or_default())
        .sum();
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && send_sizes.len() >= comm.size() as usize
            && s_in.len() >= s_total,
        comm,
    ) {
        bail!(Error::InSliceError(
            "scatterv input size @ root should be >= sum of send_sizes"
                .to_string()
        ))
    }
    let rcv_size = scatter_one(Some(send_sizes), root, comm)?;
    let o_size: usize = rcv_size.to_usize().unwrap_or_default();
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
    Ok(rcv_size)
}

pub fn validate_gatherv<T, S>(
    s_in: &[T],
    s_out: Option<&[T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[S]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<S>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let snd_size = scatter_one(recv_sizes, root, comm)?;
    let snd_usize = snd_size.to_usize().unwrap_or_default();
    let i_len = s_in.len();
    if !all_of(
        if snd_usize == 0 {
            s_in.is_empty()
        } else {
            i_len >= snd_usize
        },
        comm,
    ) {
        bail!(Error::InSliceError(format!(
            "gather input size should be atleast recv_sizes @ root: R({snd_usize}) != IN({i_len})."
        )))
    }
    let s_out = s_out.unwrap_or(&[]);
    let recv_sizes = recv_sizes.unwrap_or(&[]);
    let exp_osize = recv_sizes
        .iter()
        .map(|x| x.to_usize().unwrap_or_default())
        .sum();
    if !any_of(
        comm.rank() == root && exp_osize > 0 && exp_osize <= s_out.len(),
        comm,
    ) {
        bail!(Error::OutSliceLengthError(exp_osize, s_out.len()));
    }
    Ok(snd_size)
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

mod big;
pub use big::{
    all2all_big_vec, all2allv_big, all2allv_big_slice, all2allv_big_vec,
    all2allv_via_scatter_big, all2allv_via_scatter_big_slice,
    all2allv_via_scatter_big_vec, gatherv_big, gatherv_big_vec, scatterv_big,
    scatterv_big_vec,
};

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
    validate_scatterv(s_in, s_out, send_sizes, root, comm)?;
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
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
    validate_gatherv(
        s_in,
        s_out.as_ref().map(|x| x.as_ref()),
        recv_sizes,
        root,
        comm,
    )?;

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
    validate_all2all(a_in, a_out, comm)?;
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

fn all2allv_<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let iargs = args.to_i32();
    let send_part = Partition::new(s_in, &iargs.snd_cts[..], &iargs.snd_disp[..]);
    let mut rcv_part =
        PartitionMut::new(s_out, &iargs.rcv_cts[..], &iargs.rcv_disp[..]);
    comm.all_to_all_varcount_into(&send_part, &mut rcv_part);
    Ok(())
}

pub fn all2allv<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
    S: 'static + MCount,
{
    let uargs = args.to_usize();
    let total_send = uargs.snd_cts.iter().sum::<usize>();
    let total_rcv = uargs.rcv_cts.iter().sum::<usize>();
    let local_max = total_send.max(total_rcv);
    let g_max = allreduce(&local_max, comm, SystemOperation::max());
    //  Handle large size
    if g_max > i32::MAX as usize {
        big::all2allv_big(s_in, s_out, args, comm)
    } else {
        all2allv_(s_in, s_out, args, comm)
    }
}

pub fn all2allv_slice<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2allv(s_in, s_out, send_counts, recv_counts, comm)?;
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

fn all2allv_via_scatter_<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
    S: 'static + MCount,
{
    let iargs = args.to_i32();
    for i in 0..comm.size() {
        let rcv_start = iargs.rcv_disp[i as usize].to_usize().unwrap();
        let rcv_size = iargs.rcv_cts[i as usize].to_usize().unwrap();
        let rcv_s_out = &mut s_out[rcv_start..rcv_start + rcv_size];
        if i == comm.rank() {
            scatterv(Some(s_in), rcv_s_out, Some(&iargs.snd_cts), i, comm)?;
        } else {
            scatterv(None, rcv_s_out, None, i, comm)?;
        }
        comm.barrier();
    }
    Ok(())
}

pub fn all2allv_via_scatter<T, S>(
    s_in: &[T],
    s_out: &mut [T],
    args: &All2allvArgs<S>,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
    S: 'static + MCount,
{
    let uargs = args.to_usize();
    let total_send = uargs.snd_cts.iter().sum::<usize>();
    let total_rcv = uargs.rcv_cts.iter().sum::<usize>();
    let local_max = total_send.max(total_rcv);
    let g_max = allreduce(&local_max, comm, SystemOperation::max());
    //  Handle large size
    if g_max > i32::MAX as usize {
        todo!("Handle Big");
    } else {
        all2allv_via_scatter_(s_in, s_out, args, comm)?
    }
    Ok(())
}

pub fn all2allv_via_scatter_slice<T>(
    s_in: &[T],
    s_out: &mut [T],
    send_counts: &[usize],
    recv_counts: &[usize],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2allv(s_in, s_out, send_counts, recv_counts, comm)?;
    let params = All2allvArgs::<usize>::from_counts(send_counts, recv_counts);
    all2allv_via_scatter(s_in, s_out, &params, comm)
}

pub fn all2allv_via_scatter_vec<T>(
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
    all2allv_via_scatter_slice(
        s_in,
        &mut rcv_vec,
        send_counts,
        recv_counts,
        comm,
    )?;
    Ok(rcv_vec)
}
