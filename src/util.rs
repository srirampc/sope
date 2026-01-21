use core::slice::Iter;
use mpi::{
    datatype::{DatatypeRef, UncommittedDatatypeRef, UserDatatype},
    traits::Equivalence,
};
use num::Zero;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use std::{
    cmp::Ordering,
    marker::PhantomData,
    mem::offset_of,
    ops::{AddAssign, Mul},
    sync::atomic::AtomicPtr,
};
use typemap::{Key, ShareMap};

struct DTKeyWrapper<W>(PhantomData<W>);

impl<W: 'static> Key for DTKeyWrapper<W> {
    type Value = AtomicPtr<W>;
}

#[derive(Debug)]
pub struct Pair<T1, T2> {
    pub first: T1,
    pub second: T2,
}

impl<T1, T2> Pair<T1, T2> {
    pub fn new(first: T1, second: T2) -> Self {
        Pair { first, second }
    }

    pub fn from_tuple((first, second): (T1, T2)) -> Self {
        Pair { first, second }
    }
}

impl<T> Pair<T, T> {
    pub fn at(&self, i: usize) -> &T {
        if i == 0 { &self.first } else { &self.second }
    }

    pub fn map<B, F>(&self, mfn: F) -> Pair<B, B>
    where
        F: Fn(&T) -> B,
    {
        Pair::new(mfn(&self.first), mfn(&self.second))
    }

    pub fn zip_map<S, B, F>(&self, other: &Pair<S, S>, mfn: F) -> Pair<B, B>
    where
        F: Fn(&T, &S) -> B,
    {
        Pair::new(
            mfn(&self.first, &other.first),
            mfn(&self.second, &other.second),
        )
    }
}

impl<T1: Clone, T2: Clone> Pair<T1, T2> {
    pub fn to_tuple(&self) -> (T1, T2) {
        (self.first.clone(), self.second.clone())
    }
}

impl<T1: Clone, T2: Clone> Clone for Pair<T1, T2> {
    fn clone(&self) -> Self {
        Pair::new(self.first.clone(), self.second.clone())
    }
}

impl<T1: Default, T2: Default> Default for Pair<T1, T2> {
    fn default() -> Self {
        Pair::new(T1::default(), T2::default())
    }
}

///
///
/// This is based on the comment in Rust lang Forums:
/// https://users.rust-lang.org/t/any-way-to-create-a-generic-static/73556/2
/// code :
/// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=b7630bd5c87ae0147e099ea2bf7010e9
///
unsafe impl<'a, 'b, T1, T2> Equivalence for Pair<T1, T2>
where
    T1: Equivalence<Out = DatatypeRef<'a>> + Clone + Default,
    T2: Equivalence<Out = DatatypeRef<'b>> + Clone + Default,
    //'a : 'out,
    //'b : 'out,
{
    // TODO:: How do I note that Out's life is the shortest of 'a and 'b
    type Out = DatatypeRef<'b>;

    fn equivalent_datatype() -> Self::Out {
        static DTYPE: Lazy<Mutex<ShareMap>> =
            Lazy::new(|| Mutex::new(ShareMap::custom()));

        let mut rx = MutexGuard::map(DTYPE.lock(), |map| {
            map.entry::<DTKeyWrapper<UserDatatype>>()
                .or_insert_with(|| {
                    AtomicPtr::new(Box::into_raw(Box::new(
                        UserDatatype::structured(
                            &[1 as mpi::Count, 1 as mpi::Count],
                            &[
                                offset_of!(Pair<T1, T2>, first) as mpi::Address,
                                offset_of!(Pair<T1, T2>, second) as mpi::Address,
                            ],
                            &[
                                UncommittedDatatypeRef::from(
                                    T1::equivalent_datatype(),
                                ),
                                UncommittedDatatypeRef::from(
                                    T2::equivalent_datatype(),
                                ),
                            ],
                        ),
                    )))
                })
        });
        unsafe { rx.get_mut().as_ref().unwrap().as_ref() }
    }
}

///
/// Inclusive prefix sum (i-th sum includes i-th entry)
pub fn inc_prefix_sum<ItrT, T, SeqT>(in_itr: ItrT, scale: T) -> SeqT
where
    ItrT: Iterator<Item = T>,
    T: Zero + Mul<Output = T> + AddAssign + Clone,
    SeqT: FromIterator<T>,
{
    in_itr
        .scan(T::zero(), |state, x| {
            *state += x;
            let cstate = (*state).clone() * scale.clone();
            Some(cstate)
        })
        .collect::<SeqT>()
}

///
/// Exclusive prefix sum (i-th sum excludes i-th entry, only until i-1)
pub fn exc_prefix_sum_iter<T>(
    in_itr: Iter<'_, T>,
    scale: T,
) -> impl Iterator<Item = T>
where
    T: 'static + Zero + Mul<Output = T> + AddAssign + Clone,
{
    in_itr.scan(T::zero(), move |state, x: &T| {
        let cstate = (*state).clone() * scale.clone();
        *state += x.clone();
        Some(cstate)
    })
}

///
/// Exclusive prefix sum (i-th sum excludes i-th entry, only until i-1)
pub fn exc_prefix_sum_iterator<ItrT, T>(
    in_itr: ItrT,
    scale: T,
) -> impl Iterator<Item = T>
where
    ItrT: Iterator<Item = T>,
    T: 'static + Zero + Mul<Output = T> + AddAssign + Clone,
{
    in_itr.scan(T::zero(), move |state, x| {
        let cstate = (*state).clone() * scale.clone();
        *state += x.clone();
        Some(cstate)
    })
}

///
/// Exclusive prefix sum (i-th sum excludes i-th entry, only until i-1)
pub fn exc_prefix_sum<ItrT, T, SeqT>(in_itr: ItrT, scale: T) -> SeqT
where
    ItrT: Iterator<Item = T>,
    T: 'static + Zero + Mul<Output = T> + AddAssign + Clone,
    SeqT: FromIterator<T>,
{
    exc_prefix_sum_iterator::<ItrT, T>(in_itr, scale).collect::<SeqT>()
}

///
/// Iterator of indices
pub fn which_itr<T, F>(
    in_itr: Iter<'_, T>,
    predicate: &F,
) -> impl Iterator<Item = usize>
where
    T: 'static + Clone,
    F: Fn(&T) -> bool,
{
    in_itr
        .enumerate()
        .filter_map(|(i, x)| if predicate(x) { Some(i) } else { None })
}

///
/// Similar to C++ equal range
/// Code based on binary_search example in rust docs
/// https://doc.rust-lang.org/std/primitive.slice.html#method.binary_search
pub fn equal_range_by<T, F>(
    s: &[T],
    begin: usize,
    value: &T,
    compare: F,
) -> Pair<usize, usize>
where
    T: Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    Pair::new(
        begin + s[begin..].partition_point(|x| compare(x, value).is_lt()),
        begin + s[begin..].partition_point(|x| compare(x, value).is_le()),
    )
}

pub fn equal_range<T, F>(s: &[T], begin: usize, value: &T) -> Pair<usize, usize>
where
    T: Default + Clone + Ord,
{
    equal_range_by(s, begin, value, T::cmp)
}
