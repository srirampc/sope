//! Neighbor shift operations along the rank axis.
//!
//! These helpers move a single value or a slice between neighboring
//! processes of the communicator using non-blocking point-to-point
//! sends and receives:
//!
//! * [`right_shift`] / [`right_shift_vec`] move data from rank `r`
//!   to rank `r + 1`. After the call, rank `0` returns `None`
//!   (it received nothing); every other rank returns `Some(value)`
//!   with the data sent by its left neighbor.
//! * [`left_shift`] / [`left_shift_vec`] move data the other way:
//!   rank `r` sends to `r - 1`, the last rank returns `None`, and
//!   every other rank returns `Some(value)` with the data sent by
//!   its right neighbor.
//!
//! All four functions are collective: every rank in the
//! communicator must participate. The vector variants first run the
//! corresponding scalar shift on the slice length so that each
//! receiver allocates a buffer of the correct size before posting
//! the receive. A fixed message tag is used internally to avoid
//! collisions with unrelated point-to-point traffic.

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

use mpi::datatype::Equivalence;
use mpi::traits::{Communicator, Destination, Source};


/// Shift one element to the right neighbor.
///
/// # Description
/// Send `t` from rank `r` to rank `r + 1`. Rank `0` does not
/// receive anything (and returns `None`); every other rank returns
/// `Some(value)` with the element sent by rank `r - 1`. The last
/// rank only sends and discards the post-shift result.
///
/// # Arguments
/// * `t` - value to send to the right neighbor.
/// * `comm` - Communicator
///
/// # Returns
/// `None` on rank `0`; `Some(value_from_left_neighbor)` on every
/// other rank.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init();
/// let svalue: i32 = c.rank;
/// let rshift = right_shift(&svalue, &c.comm);
/// if c.rank == 0 {
///     assert_eq!(rshift, None);
/// } else {
///     assert_eq!(rshift, Some(c.rank - 1));
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


/// Shift one element to the left neighbor.
///
/// # Description
/// Send `t` from rank `r` to rank `r - 1`. The last rank
/// (`r == size - 1`) does not receive anything (and returns
/// `None`); every other rank returns `Some(value)` with the
/// element sent by rank `r + 1`. Rank `0` only sends and discards
/// the post-shift result.
///
/// # Arguments
/// * `t` - value to send to the left neighbor.
/// * `comm` - Communicator
///
/// # Returns
/// `None` on the last rank; `Some(value_from_right_neighbor)` on
/// every other rank.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init();
/// let svalue: i32 = c.rank;
/// let lshift = left_shift(&svalue, &c.comm);
/// if c.rank < c.size - 1 {
///     assert_eq!(lshift, Some(c.rank + 1));
/// } else {
///     assert_eq!(lshift, None);
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


/// Shift a slice to the right neighbor.
///
/// # Description
/// Send `s_in` from rank `r` to rank `r + 1`. The slice length is
/// shifted first via [`right_shift`] so that each receiver
/// allocates a buffer of the correct size before posting the
/// receive (per-rank slice lengths can differ). Rank `0` returns
/// `None`; every other rank returns `Some(vec)` with the slice
/// sent by rank `r - 1`.
///
/// # Arguments
/// * `s_in` - slice to send to the right neighbor.
/// * `comm` - Communicator
///
/// # Returns
/// `None` on rank `0`; `Some(vec_from_left_neighbor)` on every
/// other rank.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init();
/// let svalue = vec![c.rank, c.rank + c.size];
/// let rshift = right_shift_vec(&svalue, &c.comm);
/// if c.rank == 0 {
///     assert_eq!(rshift, None);
/// } else {
///     assert_eq!(rshift, Some(vec![c.rank - 1, c.rank - 1 + c.size]));
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

/// Shift a slice to the left neighbor.
///
/// # Description
/// Send `s_in` from rank `r` to rank `r - 1`. The slice length is
/// shifted first via [`left_shift`] so that each receiver
/// allocates a buffer of the correct size before posting the
/// receive (per-rank slice lengths can differ). The last rank
/// returns `None`; every other rank returns `Some(vec)` with the
/// slice sent by rank `r + 1`.
///
/// # Arguments
/// * `s_in` - slice to send to the left neighbor.
/// * `comm` - Communicator
///
/// # Returns
/// `None` on the last rank; `Some(vec_from_right_neighbor)` on
/// every other rank.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init();
/// let svalue = vec![c.rank, c.rank + c.size];
/// let lshift = left_shift_vec(&svalue, &c.comm);
/// if c.rank < c.size - 1 {
///     assert_eq!(lshift, Some(vec![c.rank + 1, c.rank + 1 + c.size]));
/// } else {
///     assert_eq!(lshift, None);
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
