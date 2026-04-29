//! Distribution schemes for a flat array spread across MPI processes.
//!
//! This module provides the [`Dist`] trait, an abstract description of
//! how a global flat array of size `n` is partitioned among `p`
//! processes, together with three concrete implementations:
//!
//! * [`ModuloDist`] - balanced block distribution where the first
//!   `n % p` ranks own one extra element.
//! * [`InterleavedDist`] - balanced block distribution where the
//!   "extra" elements are spread evenly across all ranks using the
//!   `(rank * n) / p` formula.
//! * [`ArbitDist`] - arbitrary distribution given an explicit
//!   per-rank block-sizes vector.
//!
//! For a global array `A[0..n]` distributed across processes, the
//! partition assigns to rank `r` a contiguous range of *global*
//! indices `[start_at(r) .. end_at(r))` of length `local_size_at(r)`.
//! The trait provides utilities to:
//!
//! * locate the [`Dist::owner`] of a global index,
//! * map between [`Dist::local_index`] / [`Dist::global_index`],
//! * obtain per-rank [`Dist::block_sizes`],
//! * compute [`Dist::over_under`] differences against another count
//!   distribution, useful when re-balancing data between processes.

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

use num::ToPrimitive;
use std::{iter::zip, ops::Range};

use crate::util::exc_prefix_sum_iter;

/// Abstract block distribution of a flat global array.
///
/// # Description
/// A `Dist` describes how a global array of `global_size()` elements is
/// split into contiguous *blocks* across `comm_size()` MPI processes.
/// Each implementor decides how to assign blocks to ranks; this trait
/// only fixes the abstract interface used by the rest of the library
/// (mapping global ↔ local indices, locating owners, etc.).
///
/// All ranks in the same communicator are expected to construct
/// matching `Dist` instances (same `global_size`, same scheme).
pub trait Dist {
    /// Total number of processes participating in this distribution.
    fn comm_size(&self) -> u32;

    /// Rank of *this* process within the distribution.
    fn comm_rank(&self) -> u32;

    /// Total size of the global array, i.e. the sum of all local sizes.
    fn global_size(&self) -> usize;

    /// Number of array elements held locally by this process.
    fn local_size(&self) -> usize;

    /// Number of array elements held by the process with `rank`.
    fn local_size_at(&self, rank: i32) -> usize;

    /// Process id that owns the element at global index `gidx`.
    ///
    /// `gidx` must satisfy `0 <= gidx < global_size()`.
    fn owner(&self, gidx: usize) -> i32;

    /// Global starting index of the block held on this process
    /// (inclusive lower bound of [`Dist::range`]).
    fn start(&self) -> usize;

    /// Global starting index of the block held on the process with
    /// `rank`.
    fn start_at(&self, rank: i32) -> usize;

    /// Global ending index of the block held on this process
    /// (exclusive upper bound of [`Dist::range`]).
    fn end(&self) -> usize;

    /// Global ending index of the block held on the process with
    /// `rank`.
    fn end_at(&self, rank: i32) -> usize;

    /// Half-open range `start()..end()` of global indices owned by
    /// this process.
    fn range(&self) -> Range<usize> {
        self.start()..self.end()
    }

    /// Half-open range `start_at(rank)..end_at(rank)` of global
    /// indices owned by `rank`.
    fn range_at(&self, rank: i32) -> Range<usize> {
        self.start_at(rank)..self.end_at(rank)
    }

    /// Convert a global index `gidx` into the corresponding local
    /// index on its owner process.
    fn local_index(&self, gidx: usize) -> usize {
        gidx - self.start_at(self.owner(gidx))
    }

    /// Convert a local index `lidx` on `rank` back into a global
    /// index.
    fn global_index(&self, rank: i32, lidx: usize) -> usize {
        self.start_at(rank) + lidx
    }

    /// Vector of `local_size_at(r)` for every rank `r`.
    ///
    /// Useful as the `counts` argument to MPI variable-length
    /// collectives.
    fn block_sizes(&self) -> Vec<usize> {
        (0..self.comm_size())
            .map(|x| self.local_size_at(x as i32))
            .collect()
    }

    /// Compare an external per-rank count vector against the
    /// distribution's own block sizes.
    ///
    /// # Description
    /// For every rank `r`, given an external count `counts[r]`,
    /// compute how much it is *over* or *under* the local size of
    /// this distribution. Returns a pair of vectors `(over, under)`
    /// where exactly one of `over[r]` or `under[r]` is non-zero:
    ///
    /// * `over[r] = counts[r] - local_size_at(r)` if positive,
    ///   else `0`.
    /// * `under[r] = local_size_at(r) - counts[r]` if positive,
    ///   else `0`.
    ///
    /// This is used by re-distribution code to compute how much data
    /// each rank should send/receive to match a target distribution.
    fn over_under(&self, counts: &[isize]) -> (Vec<isize>, Vec<isize>) {
        zip(0..self.comm_size() as i32, counts)
            .map(|(r, r_size)| {
                let r_local = self.local_size_at(r);
                let r_uz = r_size.to_usize().unwrap();
                let r_diff = r_local.abs_diff(r_uz);
                if r_uz > r_local {
                    (r_diff as isize, 0isize)
                } else {
                    (0isize, r_diff as isize)
                }
            })
            .unzip()
    }
}

/// Balanced block distribution: the first `n % p` ranks get one extra
/// element.
///
/// # Description
/// Given a global array of size `n` shared across `p` processes,
/// `ModuloDist` assigns:
///
/// * `ceil(n / p) = n/p + 1` elements to ranks `0 .. n%p`,
/// * `floor(n / p) = n/p` elements to ranks `n%p .. p`.
///
/// Blocks are contiguous in global index order, so rank `r`'s block
/// is `[start_at(r), end_at(r))`. All derived quantities (`_div`,
/// `_mod`, `_div1mod`, `_prefix`, `_local_size`) are pre-computed at
/// construction so that `Dist` queries run in `O(1)`.
///
/// # Examples
/// \```
/// // n = 10 elements over p = 3 processes:
/// // rank 0: [0..4)  (4 elements)
/// // rank 1: [4..7)  (3 elements)
/// // rank 2: [7..10) (3 elements)
/// let d = ModuloDist::new(10, 3, 0);
/// assert_eq!(d.local_size(), 4);
/// assert_eq!(d.range(), 0..4);
/// assert_eq!(d.owner(5), 1);
/// \```
pub struct ModuloDist {
    /// Total global array size.
    _n: usize,
    /// Number of processes.
    _comm_size: u32,
    /// Rank of this process.
    _comm_rank: u32,
    /// `n / p` (floor division).
    _div: usize,
    /// `n % p` (number of "fat" ranks holding one extra element).
    _mod: usize,
    /// Number of elements held by this process.
    _local_size: usize,
    /// Exclusive prefix: number of elements on ranks before this one.
    _prefix: usize,
    /// `(n/p + 1) * (n%p)` - end-of-block-of-fat-ranks marker, used by
    /// [`Dist::owner`].
    _div1mod: usize,
}

impl ModuloDist {
    /// Block start at `rank` for an array of `global_size` distributed
    /// over `nproc` processes (without constructing a full instance).
    pub fn block_start(global_size: usize, nproc: i32, rank: i32) -> usize {
        ModuloDist::new(global_size, nproc, rank).start()
    }

    /// Block end at `rank` for an array of `global_size` distributed
    /// over `nproc` processes (without constructing a full instance).
    pub fn block_end(global_size: usize, nproc: i32, rank: i32) -> usize {
        ModuloDist::new(global_size, nproc, rank).end()
    }

    /// Construct a `ModuloDist` for `global_size` elements shared by
    /// `comm_size` processes, from the point of view of process
    /// `comm_rank`.
    pub fn new(global_size: usize, comm_size: i32, comm_rank: i32) -> Self {
        let _comm_size: usize = comm_size as usize;
        let _comm_rank: usize = comm_rank as usize;
        let _div: usize = global_size / _comm_size;
        let _mod: usize = global_size % _comm_size;
        let _local_size: usize = _div + (if _comm_rank < _mod { 1 } else { 0 });
        let _div1mod: usize = (_div + 1) * _mod;
        let _prefix: usize = _div * _comm_rank + usize::min(_mod, _comm_rank);

        ModuloDist {
            _n: global_size,
            _comm_size: _comm_size as u32,
            _comm_rank: _comm_rank as u32,
            _div,
            _mod,
            _div1mod,
            _local_size,
            _prefix,
        }
    }
}

/// `Dist` implementation for [`ModuloDist`].
impl Dist for ModuloDist {
    fn global_size(&self) -> usize {
        self._n
    }

    fn comm_size(&self) -> u32 {
        self._comm_size
    }

    fn comm_rank(&self) -> u32 {
        self._comm_rank
    }

    fn local_size(&self) -> usize {
        self._local_size
    }

    fn local_size_at(&self, rank: i32) -> usize {
        self._div + if (rank as usize) < self._mod { 1 } else { 0 }
    }

    fn owner(&self, gidx: usize) -> i32 {
        (if gidx < self._div1mod {
            // gidx falls within the first n % p "fat" processes.
            gidx / (self._div + 1)
        } else {
            self._mod + (gidx - self._div1mod) / self._div
        }) as i32
    }

    fn end(&self) -> usize {
        self._prefix + self._local_size
    }

    fn end_at(&self, rank: i32) -> usize {
        (self._div * (rank as usize + 1))
            + usize::min(self._mod, rank as usize + 1)
    }

    fn start(&self) -> usize {
        self._prefix
    }

    fn start_at(&self, rank: i32) -> usize {
        (self._div * rank as usize) + usize::min(self._mod, rank as usize)
    }
}


/// Balanced block distribution where the extras are interleaved.
///
/// # Description
/// Same total local sizes as [`ModuloDist`] but the boundary between
/// ranks is computed as `start_at(r) = (r * n) / p`, i.e. the
/// "interleaved" formula commonly used in scientific computing.
/// As a consequence, the ranks holding `ceil(n/p)` elements are
/// spread across the range `0..p`, instead of being grouped at the
/// front. Local sizes differ by at most one between ranks.
///
/// # Examples
/// \```
/// // n = 10 elements over p = 3 processes:
/// // rank 0: [0..3)  (3 elements)
/// // rank 1: [3..6)  (3 elements)
/// // rank 2: [6..10) (4 elements)
/// let d = InterleavedDist::new(10, 3, 0);
/// assert_eq!(d.local_size(), 3);
/// assert_eq!(d.range_at(2), 6..10);
/// \```
pub struct InterleavedDist {
    _n: usize,
    _nproc: u32,
    _rank: u32,
    _local_start: usize,
    _local_end: usize,
    _local_size: usize,
}

impl InterleavedDist {
    /// Block start at `rank` (interleaved scheme) for `n` elements
    /// across `nproc` processes.
    pub fn block_start(n: usize, nproc: i32, rank: i32) -> usize {
        (rank as usize * n) / nproc as usize
    }

    /// Block end at `rank` (interleaved scheme) for `n` elements
    /// across `nproc` processes.
    pub fn block_end(n: usize, nproc: i32, rank: i32) -> usize {
        ((rank as usize + 1) * n) / nproc as usize
    }

    /// Construct an `InterleavedDist` for `n` elements shared by
    /// `nproc` processes, from the point of view of process `rank`.
    pub fn new(n: usize, nproc: i32, rank: i32) -> Self {
        let _local_start = Self::block_start(n, nproc, rank);
        let _local_end = Self::block_end(n, nproc, rank);
        let _local_size = _local_end - _local_start;
        InterleavedDist {
            _n: n,
            _nproc: nproc as u32,
            _rank: rank as u32,
            _local_start,
            _local_end,
            _local_size,
        }
    }
}


/// `Dist` implementation for [`InterleavedDist`].
impl Dist for InterleavedDist {
    fn comm_size(&self) -> u32 {
        self._nproc
    }
    fn comm_rank(&self) -> u32 {
        self._rank
    }

    fn global_size(&self) -> usize {
        self._n
    }

    fn local_size(&self) -> usize {
        self._local_size
    }
    fn local_size_at(&self, rank: i32) -> usize {
        self.end_at(rank) - self.start_at(rank)
    }

    fn owner(&self, gidx: usize) -> i32 {
        (((self._nproc as usize) * ((gidx) + 1) - 1) / (self._n)) as i32
    }

    fn start(&self) -> usize {
        self._local_start
    }

    fn start_at(&self, rank: i32) -> usize {
        (rank as usize * self._n) / self.comm_size() as usize
    }

    fn end(&self) -> usize {
        self._local_end
    }

    fn end_at(&self, rank: i32) -> usize {
        ((rank as usize + 1) * self._n) / self._nproc as usize
    }
}

/// Arbitrary block distribution given an explicit per-rank size vector.
///
/// # Description
/// `ArbitDist` is constructed from a user-provided `sizes` vector of
/// length `nproc` such that `sizes[r]` is the local block size at
/// rank `r`. The starts and ends of each block are pre-computed at
/// construction using an exclusive prefix sum, so that `Dist`
/// queries run in `O(1)`. The total `n` (global size) is supplied
/// separately and is expected to equal `sizes.iter().sum()`.
///
/// # Examples
/// \```
/// // 3 processes with sizes [2, 5, 3] over n = 10:
/// // rank 0: [0..2)  rank 1: [2..7)  rank 2: [7..10)
/// let d = ArbitDist::new(10, 3, 1, vec![2, 5, 3]);
/// assert_eq!(d.local_size(), 5);
/// assert_eq!(d.range(), 2..7);
/// \```
pub struct ArbitDist {
    _n: usize,
    _nproc: u32,
    _rank: u32,
    /// Per-rank block sizes (length = `nproc`).
    _sizes: Vec<usize>,
    /// Per-rank inclusive starts (exclusive prefix sum of `_sizes`).
    _starts: Vec<usize>,
    /// Per-rank exclusive ends (`_starts[r] + _sizes[r]`).
    _ends: Vec<usize>,
}

impl ArbitDist {
    /// Construct an `ArbitDist` from an explicit `sizes` vector.
    ///
    /// # Arguments
    /// * `n` - global array size (should equal `sizes.iter().sum()`).
    /// * `nproc` - number of processes.
    /// * `rank` - rank of this process.
    /// * `sizes` - per-rank block sizes; `sizes.len()` must equal
    ///   `nproc`.
    pub fn new(n: usize, nproc: i32, rank: i32, sizes: Vec<usize>) -> Self {
        let _starts: Vec<usize> =
            exc_prefix_sum_iter(sizes.iter(), 1usize).collect();
        let _ends: Vec<usize> = zip(sizes.iter(), _starts.iter())
            .map(|(z, s)| *z + *s)
            .collect();
        ArbitDist {
            _n: n,
            _nproc: nproc as u32,
            _rank: rank as u32,
            _sizes: sizes,
            _starts,
            _ends,
        }
    }
}


/// `Dist` implementation for [`ArbitDist`].
impl Dist for ArbitDist {
    fn comm_size(&self) -> u32 {
        self._nproc
    }
    fn comm_rank(&self) -> u32 {
        self._rank
    }

    fn global_size(&self) -> usize {
        self._n
    }

    fn local_size(&self) -> usize {
        self._sizes[self.comm_rank() as usize]
    }

    fn local_size_at(&self, rank: i32) -> usize {
        self._sizes[rank as usize]
    }

    fn owner(&self, gidx: usize) -> i32 {
        (((self._nproc as usize) * ((gidx) + 1) - 1) / (self._n)) as i32
    }

    fn start(&self) -> usize {
        self._starts[self.comm_rank() as usize]
    }

    fn start_at(&self, rank: i32) -> usize {
        self._starts[rank as usize]
    }

    fn end(&self) -> usize {
        self._ends[self.comm_rank() as usize]
    }

    fn end_at(&self, rank: i32) -> usize {
        self._ends[rank as usize]
    }
}
