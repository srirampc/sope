use mpi::datatype::Equivalence;
use mpi::traits::{Communicator, Destination, Source};

pub fn right_shift<T>(t: &T, comm: &dyn Communicator) -> T
where
    T: Equivalence + Clone + Default,
{
    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let svec: Vec<T> = vec![t.clone()];
    let mut rvec: Vec<T> = vec![Default::default()];

    mpi::request::scope(|scope| {
        let rcv_req = if rank > 0 {
            Some(
                comm.process_at_rank(rank - 1)
                    .immediate_receive_into_with_tag(scope, &mut rvec, tag),
            )
        } else {
            None
        };

        if rank < size - 1 {
            comm.process_at_rank(rank + 1).send_with_tag(&svec, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    rvec[0].clone()
}


pub fn left_shift<T>(t: &T, comm: &dyn Communicator) -> T
where
    T: Equivalence + Clone + Default,
{
    let rank = comm.rank();
    let size = comm.size();
    let tag: i32 = 13;
    let svec: Vec<T> = vec![t.clone()];
    let mut rvec: Vec<T> = vec![Default::default()];

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
            comm.process_at_rank(rank - 1).send_with_tag(&svec, tag);
        }

        if let Some(rreq) = rcv_req {
            rreq.wait_without_status();
        }
    });

    rvec[0].clone()
}
