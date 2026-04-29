//! General-purpose utilities used throughout the library.
//!
//! Contains:
//!
//! * [`Pair`] - a small two-field struct that derives [`Equivalence`]
//!   so it can be exchanged through MPI.
//! * Inclusive and exclusive prefix-sum iterators / collectors used to
//!   build offset / displacement arrays for variable-length collectives.
//! * [`which_itr`] - filter an iterator returning the indices of the
//!   elements that satisfy a predicate.
//! * [`equal_range`] / [`equal_range_by`] - C++-style binary search
//!   returning the half-open `[lower, upper)` range of equal elements
//!   in a sorted slice.

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

use core::slice::Iter;
use num::Zero;
use sope_derive::GEquivalence;
use std::{
    cmp::Ordering,
    ops::{AddAssign, Mul},
};

/// Two-field tuple-like struct usable as an MPI [`Equivalence`] type.
///
/// # Description
/// `Pair` is a generic ordered pair `(first, second)`. The
/// `GEquivalence` derive macro generates the MPI datatype description
/// so it can be sent / received as a single MPI message and used as
/// the value type of reduction operations such as
/// [`crate::reduction::max_element_by`].
///
/// # Examples
/// \```
/// let p = Pair::new(1u32, 2.0f64);
/// assert_eq!(p.first, 1);
/// assert_eq!(p.second, 2.0);
/// \```
#[derive(Debug, GEquivalence)]
pub struct Pair<T1, T2> {
    /// First component.
    pub first: T1,
    /// Second component.
    pub second: T2,
}

impl<T1, T2> Pair<T1, T2> {
    /// Construct a `Pair` from two values.
    pub fn new(first: T1, second: T2) -> Self {
        Pair { first, second }
    }

    /// Construct a `Pair` from a Rust tuple `(T1, T2)`.
    pub fn from_tuple((first, second): (T1, T2)) -> Self {
        Pair { first, second }
    }
}

impl<T> Pair<T, T> {
    /// Index access for homogeneous pairs: `0` returns `first`,
    /// anything else returns `second`.
    pub fn at(&self, i: usize) -> &T {
        if i == 0 { &self.first } else { &self.second }
    }

    /// Apply `mfn` to both components and collect the results in a
    /// new `Pair<B, B>`.
    pub fn map<B, F>(&self, mfn: F) -> Pair<B, B>
    where
        F: Fn(&T) -> B,
    {
        Pair::new(mfn(&self.first), mfn(&self.second))
    }

    /// Zip two homogeneous pairs componentwise through `mfn`.
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
    /// Convert into a Rust tuple by cloning the components.
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

/// Inclusive prefix sum (`i`-th sum includes the `i`-th entry).
///
/// # Description
/// Compute the running total of the values produced by `in_itr`,
/// multiplying every emitted prefix by the constant `scale`.
/// `scale = T::one()` gives the standard inclusive prefix sum;
/// other values are useful when assembling byte offsets from element
/// counts (e.g. `scale = size_of::<U>()`).
///
/// # Arguments
/// * `in_itr` - iterator of values to be summed.
/// * `scale` - constant multiplier applied to each emitted prefix.
///
/// # Returns
/// A collection `SeqT` of the inclusive prefix sums, in iteration order.
///
/// # Examples
/// \```
/// let v: Vec<i32> = inc_prefix_sum(vec![1, 2, 3].into_iter(), 1);
/// assert_eq!(v, vec![1, 3, 6]);
/// \```
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

/// Exclusive prefix-sum iterator over a slice.
///
/// # Description
/// Same semantics as [`exc_prefix_sum`] but operating on a slice
/// iterator and returning a lazy iterator instead of a collection.
/// The first emitted value is `0` and the last entry of the input
/// is *not* included in the last emitted prefix.
///
/// # Arguments
/// * `in_itr` - slice iterator of values.
/// * `scale` - constant multiplier applied to each emitted prefix.
///
/// # Returns
/// A lazy iterator yielding the exclusive prefix sums.
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

/// Exclusive prefix-sum iterator over an arbitrary iterator.
///
/// # Description
/// Lazy version of [`exc_prefix_sum`] that consumes any iterator
/// producing `T` and yields the exclusive running prefix multiplied by
/// `scale`.
///
/// # Returns
/// A lazy iterator yielding the exclusive prefix sums.
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

/// Exclusive prefix sum (`i`-th sum excludes the `i`-th entry).
///
/// # Description
/// Compute the prefix sum where the entry at position `i` of the
/// output is the sum of inputs at positions `0..i`. Each emitted
/// value is multiplied by `scale`. This is the typical building block
/// for `displs` arrays passed to MPI variable-length collectives.
///
/// # Arguments
/// * `in_itr` - iterator of values to be summed.
/// * `scale` - constant multiplier applied to each emitted prefix.
///
/// # Returns
/// A collection `SeqT` of the exclusive prefix sums, in iteration order.
///
/// # Examples
/// \```
/// let v: Vec<i32> = exc_prefix_sum(vec![1, 2, 3].into_iter(), 1);
/// assert_eq!(v, vec![0, 1, 3]);
/// \```
pub fn exc_prefix_sum<ItrT, T, SeqT>(in_itr: ItrT, scale: T) -> SeqT
where
    ItrT: Iterator<Item = T>,
    T: 'static + Zero + Mul<Output = T> + AddAssign + Clone,
    SeqT: FromIterator<T>,
{
    exc_prefix_sum_iterator::<ItrT, T>(in_itr, scale).collect::<SeqT>()
}

/// Iterator of indices satisfying a predicate.
///
/// # Description
/// Walk the input iterator and yield the index of every element for
/// which `predicate` returns `true`. Equivalent in spirit to
/// `iter().enumerate().filter_map(...)` but with a fixed contract.
///
/// # Arguments
/// * `in_itr` - slice iterator to scan.
/// * `predicate` - returns `true` for elements whose index should be
///   yielded.
///
/// # Returns
/// A lazy iterator over the matching indices.
///
/// # Examples
/// \```
/// let v = vec![1, 2, 3, 4];
/// let idx: Vec<usize> = which_itr(v.iter(), &|x| *x % 2 == 0).collect();
/// assert_eq!(idx, vec![1, 3]);
/// \```
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

/// Equal range search using a custom comparator.
///
/// # Description
/// Mirror of C++'s `std::equal_range`. Given a slice `s` that is
/// sorted with respect to `compare` and a starting offset `begin`,
/// return the half-open range `[lower, upper)` (as a [`Pair`]) of
/// indices whose values compare equal to `value`.
///
/// Implementation uses [`slice::partition_point`] twice and is based
/// on the `binary_search` example in the Rust standard library
/// documentation.
///
/// # Arguments
/// * `s` - sorted slice (sorted on `s[begin..]`).
/// * `begin` - index from which to start the search.
/// * `value` - value to look up.
/// * `compare` - ordering used to sort `s`.
///
/// # Returns
/// `Pair { first: lower, second: upper }` as absolute indices into
/// `s`.
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

/// Equal range search using the natural [`Ord`] of `T`.
///
/// # Description
/// Convenience wrapper around [`equal_range_by`] that uses
/// [`T::cmp`] as the comparator. `s[begin..]` must be sorted in
/// ascending order.
///
/// # Returns
/// `Pair { first: lower, second: upper }` as absolute indices into
/// `s`.
///
/// # Examples
/// \```
/// let v = vec![1, 2, 2, 3, 4];
/// let r = equal_range(&v, 0, &2);
/// assert_eq!(r.to_tuple(), (1, 3));
/// \```
pub fn equal_range<T, F>(s: &[T], begin: usize, value: &T) -> Pair<usize, usize>
where
    T: Default + Clone + Ord,
{
    equal_range_by(s, begin, value, T::cmp)
}
