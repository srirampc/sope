//! MPI reduction functions
//! Reduce, all-reduce, scan, exclusive scan, predicate reductions
//! (`all_of`, `any_of`, `none_of`, `all_same`) and min/max element
//! finding across processes.

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

use crate::{shift::right_shift, util::Pair};

use mpi::{
    collective::{SystemOperation, UserOperation},
    datatype::DatatypeRef,
    traits::{
        Communicator, CommunicatorCollectives, Equivalence, Operation, Root,
    },
};

/// Reduce one element to the root process.
///
/// # Description
/// Apply the reduction operation `op` element-wise across all processes
/// and return the result on the root process. Non-root processes receive
/// `None`.
///
/// # Arguments
/// * `x` - input value contributed by each process.
/// * `root` - root process which collects the reduced result.
/// * `comm` - Communicator
/// * `op` - reduction operation (e.g. `SystemOperation::sum()`).
///
/// # Returns
/// `Some(reduced_value)` at the root process, `None` everywhere else.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let r = reduce(&(c.rank as i32), 0, &c.comm, SystemOperation::sum());
/// if c.rank == 0 {
///     assert_eq!(r, Some((0..c.size).sum()));
/// } else {
///     assert_eq!(r, None);
/// }
/// \```
pub fn reduce<T, O>(x: &T, root: i32, comm: &dyn Communicator, op: O) -> Option<T>
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let mut tr: T = T::default();
        root_process.reduce_into_root(x, &mut tr, op);
        Some(tr)
    } else {
        root_process.reduce_into(x, op);
        None
    }
}

/// Reduce a slice element-wise to the root process.
///
/// # Description
/// Apply the reduction operation `op` element-wise on each index of `x`
/// across all processes. The slices on every process must have the same
/// length. The result is returned only on the root process.
///
/// # Arguments
/// * `x` - input slice contributed by each process.
/// * `root` - root process which collects the reduced result.
/// * `comm` - Communicator
/// * `op` - reduction operation.
///
/// # Returns
/// `Some(Vec<T>)` of the same length as `x` at the root process,
/// `None` everywhere else.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let v = vec![1, 2, 3];
/// let r = reduce_vec(&v, 0, &c.comm, SystemOperation::sum());
/// if c.rank == 0 {
///    assert_eq!(r, Some(vec![c.size, 2*c.size, 3*c.size]));
/// }
/// \```
pub fn reduce_vec<T, O>(
    x: &[T],
    root: i32,
    comm: &dyn Communicator,
    op: O,
) -> Option<Vec<T>>
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let root_process = comm.process_at_rank(root);
    if comm.rank() == root {
        let mut tr: Vec<T> = vec![T::default(); x.len()];
        root_process.reduce_into_root(x, &mut tr, op);
        Some(tr)
    } else {
        root_process.reduce_into(x, op);
        None
    }
}

/// All-reduce one element across all processes.
///
/// # Description
/// Apply the reduction operation `op` across the values contributed
/// by every process and return the reduced value to all processes.
///
/// # Arguments
/// * `x` - input value contributed by each process.
/// * `comm` - Communicator
/// * `op` - reduction operation.
///
/// # Returns
/// The reduced value, available identically on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let r = allreduce(&(c.rank as i32), &c.comm, SystemOperation::sum());
/// assert_eq!(r, (0..c.size).sum());
/// \```
pub fn allreduce<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.all_reduce_into(x, &mut tr, op);
    tr
}

/// All-reduce a slice element-wise across all processes.
///
/// # Description
/// Apply the reduction operation `op` element-wise on each index of `x`.
/// Slices must have the same length on every process. The reduced
/// vector is returned to every process.
///
/// # Arguments
/// * `x` - input slice contributed by each process.
/// * `comm` - Communicator
/// * `op` - reduction operation.
///
/// # Returns
/// Reduced `Vec<T>` of the same length as `x`, identical on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let v = vec![1, 2, 3];
/// let r = allreduce_vec(&v, &c.comm, SystemOperation::sum());
/// assert_eq!(r, vec![c.size, 2*c.size, 3*c.size]);
/// \```
pub fn allreduce_vec<T, O>(x: &[T], comm: &dyn Communicator, op: O) -> Vec<T>
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: Vec<T> = vec![T::default(); x.len()];
    comm.all_reduce_into(x, &mut tr, op);
    tr
}

/// All-reduce summation convenience wrapper.
///
/// # Description
/// Compute the global sum of `x` across every process and return the
/// result on every process. Equivalent to calling [`allreduce`] with
/// `SystemOperation::sum()`.
///
/// # Arguments
/// * `x` - input value contributed by each process.
/// * `comm` - Communicator
///
/// # Returns
/// The sum of `x` across all processes.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let r = allreduce_sum(&1i32, &c.comm);
/// assert_eq!(r, c.size);
/// \```
pub fn allreduce_sum<T>(x: &T, comm: &dyn Communicator) -> T
where
    T: Equivalence + Clone + Default,
{
    allreduce(x, comm, SystemOperation::sum())
}

/// Exclusive prefix scan across processes.
///
/// # Description
/// Compute an exclusive prefix scan of `x` over the ranks: the value at
/// rank `i` is the reduction of inputs at ranks `0..i`. The value at
/// rank `0` is the identity for the operation (here `T::default()`).
///
/// # Arguments
/// * `x` - input value contributed by each process.
/// * `comm` - Communicator
/// * `op` - reduction operation.
///
/// # Returns
/// Per-rank scanned value (excluding the contribution of the current rank).
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let r = exclusive_scan(&1i32, &c.comm, SystemOperation::sum());
/// assert_eq!(r, c.rank);
/// \```
pub fn exclusive_scan<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.exclusive_scan_into(x, &mut tr, op);
    tr
}

/// Inclusive prefix scan across processes.
///
/// # Description
/// Compute an inclusive prefix scan of `x` over the ranks: the value at
/// rank `i` is the reduction of inputs at ranks `0..=i`.
///
/// # Arguments
/// * `x` - input value contributed by each process.
/// * `comm` - Communicator
/// * `op` - reduction operation.
///
/// # Returns
/// Per-rank scanned value (including the contribution of the current rank).
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let r = scan(&1i32, &c.comm, SystemOperation::sum());
/// assert_eq!(r, c.rank + 1);
/// \```
pub fn scan<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.scan_into(x, &mut tr, op);
    tr
}

/// Test whether the predicate is true on every process.
///
/// # Description
/// Logical AND reduction over a boolean predicate. Returns `true` only
/// when `x` is `true` on every rank.
///
/// # Arguments
/// * `x` - per-process boolean value.
/// * `comm` - Communicator
///
/// # Returns
/// `true` if `x` holds on every process, `false` otherwise.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// assert!(all_of(true, &c.comm));
/// assert!(!all_of(c.rank == 0, &c.comm));
/// \```
pub fn all_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_and());
    r != 0
}

/// Test whether the predicate is false on every process.
///
/// # Description
/// Returns `true` only when `x` is `false` on every rank. Implemented
/// as a logical AND of the inputs that returns `true` when the AND is
/// zero.
///
/// # Arguments
/// * `x` - per-process boolean value.
/// * `comm` - Communicator
///
/// # Returns
/// `true` if `x` is `false` on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// assert!(none_of(false, &c.comm));
/// \```
pub fn none_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_and());
    r == 0
}

/// Test whether the predicate is true on at least one process.
///
/// # Description
/// Logical OR reduction over a boolean predicate. Returns `true` if `x`
/// is `true` on any rank.
///
/// # Arguments
/// * `x` - per-process boolean value.
/// * `comm` - Communicator
///
/// # Returns
/// `true` if `x` is `true` on any process, `false` otherwise.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// assert!(any_of(c.rank == 0, &c.comm));
/// \```
pub fn any_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_or());
    r != 0
}

/// Test whether all processes provide an equal value.
///
/// # Description
/// Returns `true` when every process supplies the same `x`. Implemented
/// by right-shifting `x` and asserting that every rank (other than
/// rank 0) sees a neighbor value equal to its own.
///
/// # Arguments
/// * `x` - per-process value to compare.
/// * `comm` - Communicator
///
/// # Returns
/// `true` if all processes have an equal `x`, `false` otherwise.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// assert!(all_same(&42i32, &c.comm));
/// assert!(!all_same(&c.rank, &c.comm));
/// \```
pub fn all_same<T>(x: &T, comm: &dyn Communicator) -> bool
where
    T: Eq + Equivalence + Clone + Default,
{
    let y = right_shift(x, comm);
    let same = (comm.rank() == 0) || (y.is_some_and(|sv| sv == *x));
    all_of(same, comm)
}

type ReductionElt<T> = Pair<i32, T>;

/// Find the optimum element across processes using a comparator.
///
/// # Description
/// Internal helper that reduces a `(rank, value)` pair using the
/// user-supplied `compare` predicate. `compare(x, y)` should return
/// `true` when `x` should be preferred over `y`. Returns the rank that
/// owned the optimum value along with the value itself.
fn optimum_element_by<T, F>(
    x: &T,
    compare: F,
    comm: &dyn Communicator,
) -> (i32, T)
where
    T: 'static + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
    F: Sync + Fn(&T, &T) -> bool, // Return true, if first element is optimum
{
    let arx = [ReductionElt::<T> {
        first: comm.rank(),
        second: x.clone(),
    }];
    let mut rcv_buff = [ReductionElt::<T>::default()];
    let max_op = UserOperation::commutative(|x, y| {
        let x: &[ReductionElt<T>] = x.downcast().unwrap();
        let y: &mut [ReductionElt<T>] = y.downcast().unwrap();
        if compare(&x[0].second, &y[0].second) {
            y[0].first = x[0].first;
            y[0].second = x[0].second.clone();
        }
    });
    comm.all_reduce_into(&arx, &mut rcv_buff, &max_op);
    (rcv_buff[0].first, rcv_buff[0].second.clone())
}

/// Find the rank holding the maximum value using a comparator.
///
/// # Description
/// Returns the `(rank, value)` pair owning the maximum element across
/// all processes. The comparator `compare(a, b)` should return `true`
/// when `a` is "greater" than `b` under the desired ordering.
///
/// # Arguments
/// * `x` - per-process candidate value.
/// * `compare` - returns `true` when its first argument is greater.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the maximum, identical on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let (r, v) = max_element_by(&c.rank, |a, b| a.gt(b), &c.comm);
/// assert_eq!(r, c.size - 1);
/// assert_eq!(v, c.size - 1);
/// \```
pub fn max_element_by<T, F>(
    x: &T,
    compare: F,
    comm: &dyn Communicator,
) -> (i32, T)
where
    T: 'static + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
    F: Sync + Fn(&T, &T) -> bool, // Returns true if first value is gt second
{
    optimum_element_by(x, compare, comm)
}

/// Find the rank holding the maximum value.
///
/// # Description
/// Returns the `(rank, value)` pair owning the maximum element across
/// all processes using the natural `Ord` ordering on `T`.
///
/// # Arguments
/// * `x` - per-process candidate value.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the maximum, identical on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let (r, v) = max_element(&c.rank, &c.comm);
/// assert_eq!((r, v), (c.size - 1, c.size - 1));
/// \```
pub fn max_element<T>(x: &T, comm: &dyn Communicator) -> (i32, T)
where
    T: 'static + Ord + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
{
    max_element_by(x, |x: &T, y: &T| x.gt(y), comm)
}

/// Find the rank that holds the global maximum across distributed slices.
///
/// # Description
/// Each process supplies its own slice `sx`. The local maximum is
/// computed and then a global maximum across processes is found. The
/// returned tuple is the `(rank, value)` of the globally maximum
/// element. If a process supplies an empty slice, `T::default()` is
/// used as its candidate value.
///
/// # Arguments
/// * `sx` - local slice contributed by the calling process.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the global maximum.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let v = vec![c.rank, c.rank + 10];
/// let (r, m) = max_element_slice(&v, &c.comm);
/// assert_eq!(r, c.size - 1);
/// \```
pub fn max_element_slice<T>(sx: &[T], comm: &dyn Communicator) -> (i32, T)
where
    T: 'static + Ord + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
{
    let dfx = T::default();
    let x = sx.iter().max().unwrap_or(&dfx);
    max_element(x, comm)
}

/// Find the rank holding the minimum value using a comparator.
///
/// # Description
/// Returns the `(rank, value)` pair owning the minimum element across
/// all processes. The comparator `compare(a, b)` should return `true`
/// when `a` is "less" than `b` under the desired ordering.
///
/// # Arguments
/// * `x` - per-process candidate value.
/// * `compare` - returns `true` when its first argument is less.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the minimum, identical on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let (r, v) = min_element_by(&c.rank, |a, b| a.lt(b), &c.comm);
/// assert_eq!((r, v), (0, 0));
/// \```
pub fn min_element_by<T, F>(
    x: &T,
    compare: F,
    comm: &dyn Communicator,
) -> (i32, T)
where
    T: 'static + Eq + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
    F: Sync + Fn(&T, &T) -> bool, // Returns true if first value is gt second
{
    optimum_element_by(x, compare, comm)
}

/// Find the rank holding the minimum value.
///
/// # Description
/// Returns the `(rank, value)` pair owning the minimum element across
/// all processes using the natural `Ord` ordering on `T`.
///
/// # Arguments
/// * `x` - per-process candidate value.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the minimum, identical on every process.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let (r, v) = min_element(&c.rank, &c.comm);
/// assert_eq!((r, v), (0, 0));
/// \```
pub fn min_element<T>(x: &T, comm: &dyn Communicator) -> (i32, T)
where
    T: 'static + Ord + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
{
    min_element_by(x, |x: &T, y: &T| x.lt(y), comm)
}

/// Find the rank that holds the global minimum across distributed slices.
///
/// # Description
/// Each process supplies its own slice `sx`. The local minimum is
/// computed and then a global minimum across processes is found. The
/// returned tuple is the `(rank, value)` of the globally minimum
/// element. If a process supplies an empty slice, `T::default()` is
/// used as its candidate value.
///
/// # Arguments
/// * `sx` - local slice contributed by the calling process.
/// * `comm` - Communicator
///
/// # Returns
/// Tuple `(rank, value)` of the global minimum.
///
/// # Examples
/// \```
/// let c = crate::comm::WorldComm::init()
/// let v = vec![c.rank, c.rank + 10];
/// let (r, m) = min_element_slice(&v, &c.comm);
/// assert_eq!((r, m), (0, 0));
/// \```
pub fn min_element_slice<T>(sx: &[T], comm: &dyn Communicator) -> (i32, T)
where
    T: 'static + Ord + Equivalence<Out = DatatypeRef<'static>> + Clone + Default,
{
    let dfx = T::default();
    let x = sx.iter().min().unwrap_or(&dfx);
    min_element(x, comm)
}

//TODO:: other reductions: scan, exclusive_scan
