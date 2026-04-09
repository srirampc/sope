//! Shift permutations functions
//! Shift a value or a vector to the left/right neighboring process.


use mpi::datatype::Equivalence;
use mpi::traits::{Communicator, Destination, Source};


/// Shift one element to the right processor.
///
/// # Description
/// Shift the given input reference of the value to the right.
/// Returns None for right process.
///
/// # Arguments
/// * `t` - input value to shift.
/// * `comm` - Communicator
///
/// # Returns
/// None at root returned value, shifted value at other proccesses
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let mut svalue: i32 = c.rank; 
/// let rshift = right_shift(&svalue, &c.comm);
/// if c.rank == 0 {
///    assert_eq!(rshift, None);
/// } else {
///    assert_eq!(rshift, Some(c.rank+1));
/// }
/// \```
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


/// Shift one element to the left processor.
///
/// # Description
/// Shift the value referred by the input to the left, returns None for 
/// last process.  
///
/// # Arguments
/// * `t` - input value to shift.
/// * `comm` - Communicator
///
/// # Returns
/// None at root returned value, shifted value at other proccesses
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let mut svalue: i32 = c.rank; 
/// let rshift = left_shift(&svalue, &c.comm);
/// if c.rank < c.size - 1 {
///    assert_eq!(rshift, Some(c.rank-1));
/// } else {
///    assert_eq!(rshift, None);
/// }
/// \```
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


/// Shift a vector to the right processor.
///
/// # Description
/// Shift slice referred by the input to the right, returns 
/// None for process '0' and Some(vec) everywhere else.  
///
/// # Arguments
/// * `s_in` - input slice to shift.
/// * `comm` - Communicator
///
/// # Returns
/// None at root returned value, shifted vector at other proccesses
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let mut svalue = vec![c.rank, c.rank+c.size]; 
/// let rshift = right_shift_vec(&svalue, &c.comm);
/// if c.rank == 0 {
///    assert_eq!(rshift, None);
/// } else {
///    assert_eq!(rshift, Some(vec![c.rank+1, c.rank+1+c.size]));
/// }
/// \```
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

/// Shift a vector to the left processor.
///
/// # Description
/// Shift slice referred by the input to the left, returns 
/// None for last process and Some(vec) everywhere else.  
///
/// # Arguments
/// * `s_in` - input slice to shift.
/// * `comm` - Communicator
///
/// # Returns
/// None at root returned value, shifted value at other proccesses
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let mut svalue = vec![c.rank, c.rank+c.size]; 
/// let rshift = left_shift_vec(&svalue, &c.comm);
/// if c.rank < c.size - 1 {
///    assert_eq!(rshift, Some(vec![c.rank-1, c.rank-1+c.size]));
/// } else {
///    assert_eq!(rshift, None);
/// }
/// \```
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
