use anyhow::{Ok, Result, bail};
use mpi::traits::{Communicator, Destination, Equivalence, Source};

use crate::collective::{All2allvArgs, MCount};
use crate::reduction::any_of;
use crate::util::exc_prefix_sum;
use crate::{
    collective::{Error as CollError, scatter_one},
    reduction::all_of,
};

pub fn scatterv_big<T>(
    s_in: Option<&[T]>,
    s_out: &mut [T], // Assuming s_out has enough size to accept data
    send_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone,
{
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    if !any_of(
        comm.rank() == root
            && !s_in.is_empty()
            && send_sizes.len() >= comm.size() as usize
            && s_in.len() >= send_sizes.iter().sum::<usize>(),
        comm,
    ) {
        bail!(CollError::InSliceError(
            "scatterv input size @ root should be >= sum of send_sizes"
                .to_string()
        ))
    }

    let o_size = scatter_one(Some(send_sizes), root, comm)?;
    if !all_of(
        if o_size == 0 {
            s_out.is_empty()
        } else {
            s_out.len() >= o_size
        },
        comm,
    ) {
        bail!(CollError::OutSliceLengthError(o_size, s_out.len()));
    }

    let send_offsets: Vec<usize> =
        exc_prefix_sum(send_sizes.iter().cloned(), 1usize);

    // TODO:: send with tag?
    mpi::request::multiple_scope(comm.size() as usize, |scope, coll| {
        if comm.rank() == root {
            for i in 0..comm.size() {
                let offset = send_offsets[i as usize];
                let st = offset..(offset + send_sizes[i as usize]);
                if i != root {
                    // Do an immediate send
                    let dest_process = comm.process_at_rank(i);
                    let req = dest_process.immediate_send(scope, &s_in[st]);
                    coll.add(req);
                }
            }
        } else {
            let root_process = comm.process_at_rank(root);
            let req = root_process.immediate_receive_into(scope, &mut s_out[..]);
            coll.add(req);
        }

        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    if comm.rank() == root && send_sizes[root as usize] > 0 {
        // directly copy to output
        let offset = send_offsets[root as usize];
        let st = offset..(offset + send_sizes[root as usize]);
        s_out.clone_from_slice(&s_in[st]);
    }
    Ok(())
}

pub fn scatterv_big_vec<T>(
    s_in: Option<&[T]>,
    send_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let rcv_size = scatter_one(send_sizes, root, comm)? as usize;
    let mut rcv_vec = vec![T::default(); rcv_size];
    scatterv_big(s_in, &mut rcv_vec, send_sizes, root, comm)?;
    Ok(rcv_vec)
}

pub fn gatherv_big<T>(
    s_in: &[T],
    s_out: Option<&mut [T]>, // Assuming s_out has enough size to accept data
    recv_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    let s_len = scatter_one(recv_sizes, root, comm)?;
    let i_len = s_in.len();
    if !all_of(
        if s_len == 0 {
            s_in.is_empty()
        } else {
            i_len >= s_len
        },
        comm,
    ) {
        bail!(CollError::InSliceError(format!(
            "gather input size should be atleast recv_sizes @ root: R({s_len}) != IN({i_len})."
        )))
    }

    let s_out = s_out.unwrap_or(&mut []);
    let recv_sizes = recv_sizes.unwrap_or(&[]);
    let exp_osize = recv_sizes.iter().sum::<usize>();
    if !any_of(
        comm.rank() == root && exp_osize > 0 && exp_osize <= s_out.len(),
        comm,
    ) {
        bail!(CollError::OutSliceLengthError(exp_osize, s_out.len()));
    }

    let mut rcv_buff: Vec<Vec<T>> = if comm.rank() == root {
        recv_sizes
            .iter()
            .map(|rcz| vec![T::default(); *rcz])
            .collect()
    } else {
        vec![]
    };

    // TODO:: send with tag?
    mpi::request::multiple_scope(comm.size() as usize, |scope, coll| {
        if comm.rank() == root {
            //recivers
            for (ui, s_rcv_buf) in rcv_buff.iter_mut().enumerate() {
                if ui as i32 != comm.rank() {
                    let snd_process = comm.process_at_rank(ui as i32);
                    let req = snd_process
                        .immediate_receive_into(scope, &mut s_rcv_buf[..]);
                    coll.add(req);
                }
            }
        } else {
            let root_process = comm.process_at_rank(root);
            let req = root_process.immediate_send(scope, s_in);
            coll.add(req);
        }

        //
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // copy from buffer
    if comm.rank() == root {
        let mut rcv_offset = 0;
        for i in 0..comm.size() {
            let ui = i as usize;
            let r_range = rcv_offset..(rcv_offset + recv_sizes[ui]);
            if i != comm.rank() {
                s_out[r_range].clone_from_slice(&rcv_buff[ui]);
            } else {
                // directly copy to output
                s_out[r_range.clone()].clone_from_slice(&s_in[r_range]);
            }
            rcv_offset += recv_sizes[ui];
        }
    }
    Ok(())
}

pub fn gatherv_big_vec<T>(
    s_in: &[T],
    recv_sizes: Option<&[usize]>,
    root: i32,
    comm: &dyn Communicator,
) -> Result<Option<Vec<T>>>
where
    T: Equivalence + Default + Clone,
{
    if comm.rank() == root {
        let recv_sizes = recv_sizes.unwrap_or(&[]);
        let mut out_vec = vec![T::default(); recv_sizes.iter().sum::<usize>()];
        gatherv_big(s_in, Some(&mut out_vec), Some(recv_sizes), root, comm)?;
        Ok(Some(out_vec))
    } else {
        gatherv_big(s_in, None, None, root, comm)?;
        Ok(None)
    }
}

pub fn all2allv_big<T, S>(
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
    let mut rcv_buff: Vec<Vec<T>> = (0..comm.size())
        .map(|i| vec![T::default(); uargs.rcv_cts[i as usize]])
        .collect();
    mpi::request::multiple_scope(2 * comm.size() as usize, |scope, coll| {
        //senders
        for i in 0..comm.size() {
            if i != comm.rank() {
                let ui = i as usize;
                let snd_offset = uargs.snd_disp[ui];
                let s_range = snd_offset..(snd_offset + uargs.snd_cts[ui]);
                // Do an immediate send
                let dest_process = comm.process_at_rank(i);
                let req = dest_process.immediate_send(scope, &s_in[s_range]);
                coll.add(req);
            }
        }

        //recivers
        for (ui, s_rcv_buf) in rcv_buff.iter_mut().enumerate() {
            if ui != comm.rank() as usize {
                let snd_process = comm.process_at_rank(ui as i32);
                let req =
                    snd_process.immediate_receive_into(scope, &mut s_rcv_buf[..]);
                coll.add(req);
            }
        }
        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // copy to output slice
    for i in 0..comm.size() {
        let ui = i as usize;
        let rcv_offset = uargs.rcv_disp[ui];
        let r_range = rcv_offset..(rcv_offset + uargs.rcv_cts[ui]);
        if i != comm.rank() {
            s_out[r_range].clone_from_slice(&rcv_buff[ui]);
        } else {
            // directly copy to output
            let snd_offset = uargs.snd_disp[ui];
            let s_range = snd_offset..(snd_offset + uargs.snd_cts[ui]);
            s_out[r_range].clone_from_slice(&s_in[s_range]);
        }
    }
    Ok(())
}

pub fn all2allv_big_slice<T>(
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
        bail!(CollError::InSliceError(
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
        bail!(CollError::OutSliceLengthError(recv_total, s_out.len()));
    }

    let params = All2allvArgs::<usize>::from_counts(send_counts, recv_counts);
    all2allv_big(s_in, s_out, &params, comm)
}

pub fn all2allv_big_vec<T>(
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
    all2allv_big_slice(s_in, &mut rcv_vec, send_counts, recv_counts, comm)?;
    Ok(rcv_vec)
}
