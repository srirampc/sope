use core::slice::Iter;
use num::Zero;
use sope_derive::GEquivalence;
use std::{
    cmp::Ordering,
    ops::{AddAssign, Mul},
};

#[derive(Debug, GEquivalence)]
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
