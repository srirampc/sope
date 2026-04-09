use num::ToPrimitive;
use std::{iter::zip, ops::Range};

use crate::util::exc_prefix_sum_iter;

/// Trait representing a block distribution of a flat array
pub trait Dist {
    /// Total number of proceses in this distribution
    fn comm_size(&self) -> u32;

    /// Rank of this process
    fn comm_rank(&self) -> u32;

    /// Total size of the array spread across comm_size() processes
    fn global_size(&self) -> usize;

    /// Size of the partition of the array, locally available at this process
    fn local_size(&self) -> usize;

    /// Size of the partition of the array, locally available at rank
    fn local_size_at(&self, rank: i32) -> usize;

    /// Process id that owns the element at gidx (0 <= gidx < global_size)
    fn owner(&self, gidx: usize) -> i32;

    /// Global starting index at this processor
    fn start(&self) -> usize;

    /// Global starting index at the processor with 'rank'
    fn start_at(&self, rank: i32) -> usize;

    /// Global ending index at this processor
    fn end(&self) -> usize;

    /// Global ending index at the processor with 'rank'
    fn end_at(&self, rank: i32) -> usize;

    /// Range of global indices available at this processor
    fn range(&self) -> Range<usize> {
        self.start()..self.end()
    }

    /// Range of elements at this processor at rank
    fn range_at(&self, rank: i32) -> Range<usize> {
        self.start_at(rank)..self.end_at(rank)
    }

    /// Mapping between local index and global index
    fn local_index(&self, gidx: usize) -> usize {
        gidx - self.start_at(self.owner(gidx))
    }

    /// Global index
    fn global_index(&self, rank: i32, lidx: usize) -> usize {
        self.start_at(rank) + lidx
    }

    /// Block sizes
    fn block_sizes(&self) -> Vec<usize> {
        (0..self.comm_size())
            .map(|x| self.local_size_at(x as i32))
            .collect()
    }

    /// Over/Under correspding to the input counts
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

/// Equal Array Block Distribution with remaining elements 
/// distributed the first k processes
pub struct ModuloDist {
    /// Total global size
    _n: usize,
    _comm_size: u32,
    _comm_rank: u32,
    /// derived/buffered values (for faster computation of results)
    _div: usize, // n/p
    _mod: usize, // n%p
    /// local size (number of local elements)
    _local_size: usize,
    /// the exclusive prefix (number of elements on previous processors)
    _prefix: usize,
    /// number of elements on processors with one more element
    _div1mod: usize, // (n/p + 1)*(n % p)
}

/// Implementation of ModuloDist
impl ModuloDist {
    /// Block start at 'rank' for the given global_size
    pub fn block_start(global_size: usize, nproc: i32, rank: i32) -> usize {
        ModuloDist::new(global_size, nproc, rank).start()
    }

    /// Block end at 'rank' for the given global_size
    pub fn block_end(global_size: usize, nproc: i32, rank: i32) -> usize {
        ModuloDist::new(global_size, nproc, rank).end()
    }

    /// Constructor for ModuloDist
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

/// Dist implemenation for ModuloDist
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
            // a_i is within the first n % p processors
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


/// Equal Array Block Distribution with remaining elements 
/// distributed interleaved between the processes
pub struct InterleavedDist {
    _n: usize,
    _nproc: u32,
    _rank: u32,
    _local_start: usize,
    _local_end: usize,
    _local_size: usize,
}

impl InterleavedDist {
    pub fn block_start(n: usize, nproc: i32, rank: i32) -> usize {
        (rank as usize * n) / nproc as usize
    }

    pub fn block_end(n: usize, nproc: i32, rank: i32) -> usize {
        ((rank as usize + 1) * n) / nproc as usize
    }

    /// Constructor for InterleavedDist
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


/// Dist implemenation for InterleavedDist
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

/// Arbitrary Distribution of arrays given the block sizes assigned for each 
/// process.
pub struct ArbitDist {
    _n: usize,
    _nproc: u32,
    _rank: u32,
    _sizes: Vec<usize>,
    _starts: Vec<usize>,
    _ends: Vec<usize>,
}

impl ArbitDist {
    /// Constructor for ArbitDist
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


/// Dist implemenation for ArbitDist
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
