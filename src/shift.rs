use mpi::datatype::Equivalence;
use mpi::traits::{Communicator, Destination, Source};

pub fn right_shift<T>(t: &T, comm: &dyn Communicator) -> Option<T>
where
    T: Equivalence + Clone + Default,
{
    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let s_in = t.clone();
    let mut s_out = T::default();

    mpi::request::scope(|scope| {
        let rcv_req = if rank > 0 {
            Some(
                comm.process_at_rank(rank - 1)
                    .immediate_receive_into_with_tag(scope, &mut s_out, tag),
            )
        } else {
            None
        };

        if rank < size - 1 {
            comm.process_at_rank(rank + 1).send_with_tag(&s_in, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    if rank > 0 { Some(s_out) } else { None }
}


pub fn left_shift<T>(t: &T, comm: &dyn Communicator) -> Option<T>
where
    T: Equivalence + Clone + Default,
{
    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let s_in = t.clone();
    let mut s_out = T::default();

    mpi::request::scope(|scope| {
        let rcv_req = if rank < size - 1 {
            Some(
                comm.process_at_rank(rank + 1)
                    .immediate_receive_into_with_tag(scope, &mut s_out, tag),
            )
        } else {
            None
        };

        if rank > 0 {
            comm.process_at_rank(rank - 1).send_with_tag(&s_in, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    if rank < size - 1 { Some(s_out) } else { None }
}


pub fn right_shift_vec<T>(s_in: &[T], comm: &dyn Communicator) -> Option<Vec<T>>
where
    T: Equivalence + Clone + Default,
{
    let nrcv = right_shift(&s_in.len(), comm).unwrap_or_default();
    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let mut s_out = vec![T::default(); nrcv];

    mpi::request::scope(|scope| {
        let rcv_req = if rank > 0 {
            Some(
                comm.process_at_rank(rank - 1)
                    .immediate_receive_into_with_tag(scope, &mut s_out, tag),
            )
        } else {
            None
        };

        if rank < size - 1 {
            comm.process_at_rank(rank + 1).send_with_tag(s_in, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    if rank > 0 { Some(s_out) } else { None }
}

pub fn left_shift_vec<T>(s_in: &[T], comm: &dyn Communicator) -> Option<Vec<T>>
where
    T: Equivalence + Clone + Default,
{
    let nrcv = left_shift(&s_in.len(), comm).unwrap_or_default();

    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let mut rvec: Vec<T> = vec![T::default(); nrcv];

    mpi::request::scope(|scope| {
        let rcv_req = if rank < size - 1 {
            Some(
                comm.process_at_rank(rank + 1)
                    .immediate_receive_into_with_tag(scope, &mut rvec, tag),
            )
        } else {
            None
        };

        if rank > 0 {
            comm.process_at_rank(rank - 1).send_with_tag(s_in, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    if rank < size - 1 { Some(rvec) } else { None }
}
