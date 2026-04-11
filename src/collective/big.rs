use anyhow::{Ok, Result};
use mpi::traits::{
    Communicator, CommunicatorCollectives, Destination, Equivalence, Source,
};
use std::iter::zip;

use crate::{All2allvArgs, MCount, util::exc_prefix_sum};

use super::{
    scatter_one, validate_all2all, validate_all2allv, validate_gatherv,
    validate_scatterv,
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
    let rcv_size = validate_scatterv(s_in, s_out, send_sizes, root, comm)?;
    let s_in = s_in.unwrap_or(&[]);
    let send_sizes = send_sizes.unwrap_or(&[]);
    let send_offsets: Vec<usize> =
        exc_prefix_sum(send_sizes.iter().cloned(), 1usize);

    // TODO:: send with tag?
    mpi::request::multiple_scope(comm.size() as usize, |scope, coll| {
        if comm.rank() == root {
            for (iu, (s_size, s_offset)) in
                zip(send_sizes.iter(), send_offsets.iter()).enumerate()
            {
                let i = iu as i32;
                if i == root || *s_size == 0 {
                    continue;
                }
                // Do an immediate send to everyone but root
                let st = *s_offset..(*s_offset + *s_size);
                let dest_process = comm.process_at_rank(i);
                let req = dest_process.immediate_send(scope, &s_in[st]);
                coll.add(req);
            }
        } else if rcv_size > 0 {
            // immediate recieve from everyone
            let root_process = comm.process_at_rank(root);
            let req = root_process.immediate_receive_into(scope, &mut s_out[..]);
            coll.add(req);
        }

        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // Sending to self
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
    let snd_size = validate_gatherv(
        s_in,
        s_out.as_ref().map(|x| x.as_ref()),
        recv_sizes,
        root,
        comm,
    )?;
    let s_out = s_out.unwrap_or(&mut []);
    let recv_sizes = recv_sizes.unwrap_or(&[]);

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
                if ui as i32 == comm.rank() || s_rcv_buf.is_empty() {
                    continue;
                }
                let snd_process = comm.process_at_rank(ui as i32);
                let req =
                    snd_process.immediate_receive_into(scope, &mut s_rcv_buf[..]);
                coll.add(req);
            }
        } else if snd_size > 0 {
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
            let r_size = recv_sizes[ui];
            if r_size == 0 {
                continue;
            }
            let r_range = rcv_offset..(rcv_offset + r_size);
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

pub fn all2all_big<T>(
    a_in: &[T],
    a_out: &mut [T],
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Clone + Default,
{
    validate_all2all(a_in, a_out, comm)?;
    // n elements to recieve per processor
    let npp = a_in.len() / (comm.size() as usize);
    let mut rcv_buff: Vec<Vec<T>> =
        (0..comm.size()).map(|_i| vec![T::default(); npp]).collect();

    mpi::request::multiple_scope(2 * comm.size() as usize, |scope, coll| {
        //senders
        for i in 0..comm.size() {
            if i == comm.rank() {
                continue;
            }
            let ui = i as usize;
            let dest_process = comm.process_at_rank(i);
            let snd_offset = ui * npp;
            let s_range = snd_offset..(snd_offset + npp);
            let req = dest_process.immediate_send(scope, &a_in[s_range]);
            coll.add(req);
        }
        for (ui, s_rcv_buf) in rcv_buff.iter_mut().enumerate() {
            let i = ui as i32;
            if i == comm.rank() {
                continue;
            }
            let src_proc = comm.process_at_rank(i);
            let req = src_proc.immediate_receive_into(scope, &mut s_rcv_buf[..]);
            coll.add(req);
        }
        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    for i in 0..comm.size() {
        let ui = i as usize;
        let rcv_offset = ui * npp;
        let r_range = rcv_offset..(rcv_offset + npp);
        if i != comm.rank() {
            a_out[r_range].clone_from_slice(&rcv_buff[ui]);
        } else {
            // directly copy to output
            let snd_offset = ui * npp;
            let s_range = snd_offset..(snd_offset + npp);
            a_out[r_range].clone_from_slice(&a_in[s_range]);
        }
    }

    Ok(())
}

pub fn all2all_big_vec<T>(a_in: &[T], comm: &dyn Communicator) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
{
    let mut recv_buf: Vec<T> = vec![T::default(); a_in.len()];
    all2all_big(a_in, &mut recv_buf, comm)?;
    Ok(recv_buf)
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
        for (ui, (snd_offset, snd_count)) in
            zip(uargs.snd_disp.iter(), uargs.snd_cts.iter()).enumerate()
        {
            let i = ui as i32;
            if i == comm.rank() || *snd_count == 0 {
                continue;
            }
            let s_range = *snd_offset..(*snd_offset + *snd_count);
            // Do an immediate send
            let dest_process = comm.process_at_rank(i);
            let req = dest_process.immediate_send(scope, &s_in[s_range]);
            coll.add(req);
        }

        //recivers
        for (ui, (s_rcv_buf, rcv_count)) in
            zip(rcv_buff.iter_mut(), uargs.rcv_cts.iter()).enumerate()
        {
            let i = ui as i32;
            if i == comm.rank() || *rcv_count == 0 {
                continue;
            }
            let snd_process = comm.process_at_rank(i);
            let req =
                snd_process.immediate_receive_into(scope, &mut s_rcv_buf[..]);
            coll.add(req);
        }
        // Wait for all of them to complete
        let mut result = vec![];
        coll.wait_all(&mut result);
    });

    // copy to output slice
    for i in 0..comm.size() {
        let ui = i as usize;
        if uargs.rcv_cts[ui] == 0 {
            continue;
        }
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
    validate_all2allv(s_in, s_out, send_counts, recv_counts, comm)?;
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

pub fn all2allv_via_scatter_big<T, S>(
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
    for (ui, (rcv_start, rcv_size)) in
        zip(args.rcv_disp.iter(), args.rcv_cts.iter()).enumerate()
    {
        let i = ui as i32;
        let rcv_start = rcv_start.to_usize().unwrap();
        let rcv_size = rcv_size.to_usize().unwrap();
        let rcv_s_out = &mut s_out[rcv_start..rcv_start + rcv_size];
        if i == comm.rank() {
            scatterv_big(Some(s_in), rcv_s_out, Some(&uargs.snd_cts), i, comm)?;
        } else {
            scatterv_big(None, rcv_s_out, None, i, comm)?;
        }
        comm.barrier();
    }
    Ok(())
}

pub fn all2allv_via_scatter_big_slice<T>(
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
    all2allv_via_scatter_big(s_in, s_out, &params, comm)
}

pub fn all2allv_via_scatter_big_vec<T>(
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
    all2allv_via_scatter_big_slice(
        s_in,
        &mut rcv_vec,
        send_counts,
        recv_counts,
        comm,
    )?;
    Ok(rcv_vec)
}
