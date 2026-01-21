use crate::{shift::right_shift, util::Pair};

use mpi::{
    collective::{SystemOperation, UserOperation},
    datatype::DatatypeRef,
    traits::{Communicator, CommunicatorCollectives, Equivalence, Operation, Root},
};

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

pub fn reduce_vec<T, O>(x: &[T], root: i32, comm: &dyn Communicator, op: O) -> Option<Vec<T>>
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

pub fn allreduce<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.all_reduce_into(x, &mut tr, op);
    tr
}

pub fn allreduce_vec<T, O>(x: &[T], comm: &dyn Communicator, op: O) -> Vec<T>
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: Vec<T> = vec![T::default(); x.len()];
    comm.all_reduce_into(x, &mut tr, op);
    tr
}

pub fn allreduce_sum<T>(x: &T, comm: &dyn Communicator) -> T
where
    T: Equivalence + Clone + Default,
{
    allreduce(x, comm, SystemOperation::sum())
}

pub fn exclusive_scan<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.exclusive_scan_into(x, &mut tr, op);
    tr
}

pub fn scan<T, O>(x: &T, comm: &dyn Communicator, op: O) -> T
where
    T: Equivalence + Clone + Default,
    O: Operation,
{
    let mut tr: T = T::default();
    comm.scan_into(x, &mut tr, op);
    tr
}

pub fn all_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_and());
    r != 0
}

pub fn none_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_and());
    r == 0
}

pub fn any_of(x: bool, comm: &dyn Communicator) -> bool {
    let mut r: i32 = 0;
    comm.all_reduce_into(&(x as i32), &mut r, SystemOperation::logical_or());
    r != 0
}

pub fn all_same<T>(x: &T, comm: &dyn Communicator) -> bool
where
    T: Eq + Equivalence + Clone + Default,
{
    let y = right_shift(x, comm);
    let same = (comm.rank() == 0) || (y == *x);
    all_of(same, comm)
}

type ReductionElt<T> = Pair<i32, T>;

fn optimum_element_by<'a, T, F>(x: &T, compare: F, comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
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

///
/// Returns the process id with the maximum element based  on the comparator function
pub fn max_element_by<'a, T, F>(x: &T, compare: F, comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
    F: Sync + Fn(&T, &T) -> bool, // Returns true if first value is gt second
{
    optimum_element_by(x, compare, comm)
}

///
/// Returns the process id with the maximum element
pub fn max_element<'a, T>(x: &T, comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + PartialOrd + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
{
    max_element_by(x, |x: &T, y: &T| x.gt(y), comm)
}

pub fn max_element_slice<'a, T>(sx: &[T], comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + Ord + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
{
    let dfx = T::default();
    let x = sx.iter().max().unwrap_or(&dfx);
    max_element(x, comm)
}

///
/// Returns the process id with the minimum element based  on the comparator function
pub fn min_element_by<'a, T, F>(x: &T, compare: F, comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
    F: Sync + Fn(&T, &T) -> bool, // Returns true if first value is gt second
{
    optimum_element_by(x, compare, comm)
}

///
/// Returns the process id with the minimum element
pub fn min_element<'a, T>(x: &T, comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + PartialOrd + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
{
    min_element_by(x, |x: &T, y: &T| x.lt(y), comm)
}

pub fn min_element_slice<'a, T>(sx: &[T], comm: &dyn Communicator) -> (i32, T)
where
    T: Eq + Ord + Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
{
    let dfx = T::default();
    let x = sx.iter().min().unwrap_or(&dfx);
    min_element(x, comm)
}

//TODO:: global reduce functions: reduce, scan, exclusive_scan
