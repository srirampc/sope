use anyhow::{Ok, Result, bail};
use mpi::{topology::Communicator, traits::Equivalence};
use std::cmp::Ordering;

use super::Error;
use crate::{
    collective::{all2all_vec, all2allv_vec, allgatherv_vec},
    distribution::{arbit_distribute, distribute_vec, stable_distribute},
    partition::{Dist, ModuloDist},
    reduction::{all_of, allreduce_sum},
    sort::bitonicsort::bitonic_sort,
    util::equal_range_by,
};

pub fn sample_arbit_decomp<T, F>(
    t_in: &[T],
    compare: F,
    s: usize,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let local_size: usize = t_in.len();
    let total_size: usize = allreduce_sum(&local_size, comm);
    let p = comm.size();
    let total_s = s * p as usize;

    // pick s*p samples, i.e. max(ceil((local_size/n)*s*p), 1) locally
    let local_s = if local_size == 0 {
        0
    } else {
        //1usize.max(((local_size * total_s) + total_size - 1) / total_size)
        1usize.max((local_size * total_s).div_ceil(total_size))
    };

    if !all_of(allreduce_sum(&local_size, comm) >= total_s, comm) {
        bail!(Error::SampleSizeError)
    }

    //  pick local samples if (local_s > 0)
    let local_splitters = if local_s > 0 {
        let mut l_splitters: Vec<T> = vec![T::default(); local_s];
        let mut pos: usize = 0;
        for (i, lx) in l_splitters.iter_mut().enumerate() {
            // modulo-based bucket size
            let bucket_size: usize = local_size / (local_s + 1)
                + (i < (local_size % (local_s + 1))) as usize;
            // pick last element of each bucket
            pos += bucket_size - 1;
            *lx = t_in[pos].clone();
            pos += 1;
        }
        l_splitters
    } else {
        vec![]
    };

    //distribute elements equally
    // TODO:: distribute_inplace instaed ?
    let s_part = ModuloDist::new(
        allreduce_sum(&local_splitters.len(), comm),
        p,
        comm.rank(),
    );
    let mut local_splitters = distribute_vec(&local_splitters, &s_part, comm)?;

    // Should have atleast s splitters
    if !all_of(local_splitters.len() >= s, comm) {
        bail!(Error::SplitterSizeError(
            "Number of splitters less than expected.".to_string()
        ));
    }

    // discard extra splitters, to make it even
    if local_splitters.len() != s {
        local_splitters.resize(s, T::default());
    }

    //  sort splitters using parallel bitonic sort
    bitonic_sort(&mut local_splitters, compare, comm)?;

    // select the last element on each process but the last
    let my_splitter: T = local_splitters
        .last()
        .ok_or(Error::MissingLastError)?
        .clone();

    // allgather splitters (from all but the last processor)
    let mut recv_sizes: Vec<i32> = vec![1; comm.size() as usize];
    recv_sizes[comm.size() as usize - 1] = 0;
    let sv = if comm.rank() != comm.size() - 1 {
        vec![my_splitter]
    } else {
        vec![]
    };
    allgatherv_vec(&sv, &recv_sizes, comm)
}

pub fn sample_block_decomp<T, F>(
    t_in: &mut [T],
    compare: F,
    s: usize,
    comm: &dyn Communicator,
) -> Result<Vec<T>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    let local_size: usize = t_in.len();
    if !all_of(local_size > 0, comm) {
        bail!(Error::SampleSizeError)
    }

    // samples
    //  - pick `s` samples equally spaced such that `s` samples define `s+1`
    //    subsequences in the sorted order
    let bucket_f: usize = local_size / (s + 1);
    let mut local_splitters: Vec<T> = vec![T::default(); s];
    let mut pos: usize = 0;
    for (i, lx) in local_splitters.iter_mut().enumerate() {
        let bucket_size = bucket_f + (i < (local_size % (s + 1))) as usize;
        pos += bucket_size - 1;
        *lx = t_in[pos].clone();
        pos += 1;
    }

    // sort splitters using parallel bitonic sort
    bitonic_sort(&mut local_splitters[..], compare, comm)?;

    // select the last element on each process but the last
    let my_splitter: T = local_splitters
        .last()
        .ok_or(Error::MissingLastError)?
        .clone();

    // allgather splitters (from all but the last processor)
    let mut recv_sizes: Vec<i32> = vec![1; comm.size() as usize];
    recv_sizes[comm.size() as usize - 1] = 0;
    let sv = if comm.rank() != comm.size() - 1 {
        vec![my_splitter]
    } else {
        vec![]
    };
    let result_splitters = allgatherv_vec(&sv, &recv_sizes[..], comm)?;
    Ok(result_splitters)
}

fn split<T, F>(
    tsl: &mut [T],
    splitters: &[T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<Vec<usize>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if !all_of(splitters.len() == comm.size() as usize - 1, comm) {
        bail!(Error::SplitterSizeError("Size not p - 1.".to_string()));
    }
    // Locally find splitter positions in data
    //  (if an identical splitter appears at least three times (or more),
    //  then split the intermediary buckets evenly) => send_counts

    let local_size = tsl.len();
    let local_part = ModuloDist::new(local_size, comm.size(), comm.rank());
    let mut send_counts: Vec<usize> = vec![0; comm.size() as usize];
    let mut pos: usize = 0;
    let mut i: usize = 0;

    while i < splitters.len() {
        // get the number of splitters which are equal starting from `i`
        let mut split_by: usize = 1;
        while i + split_by < splitters.len()
            && !compare(&splitters[i], &splitters[i + split_by]).is_lt()
        {
            split_by += 1;
        }
        // get the range of equal elements
        let eqr = equal_range_by(tsl, pos, &splitters[i], &compare);
        // assign smaller elements to processor left of splitter (= `i`)
        send_counts[i] += eqr.first - pos;
        pos = eqr.first;
        // split equal elements fairly across processors
        let mut eq_size = eqr.second - pos;
        //  - try to split approx equal:
        let eq_size_split = (eq_size + send_counts[i]) / (split_by + 1) + 1;
        for j in 0..split_by {
            let mut out_size: usize = 0;
            let lpart_size = local_part.local_size_at((i + j) as i32);
            if send_counts[i + j] < lpart_size {
                out_size = usize::min(
                    usize::max(lpart_size - send_counts[i + j], eq_size_split),
                    eq_size,
                );
                eq_size -= out_size;
            }
            send_counts[i + j] += out_size;
        }
        // - assign remaining elements to next processor
        send_counts[i + split_by] += eq_size;
        i += split_by;
        pos = eqr.second;
    }
    // send last elements to last processor
    let out_size = tsl.len() - pos;
    send_counts[comm.size() as usize - 1] += out_size;
    // variant check
    if !all_of(send_counts.iter().sum::<usize>() == local_size, comm) {
        bail!(Error::SortInvariantError(
            "send_counts.iter().sum() === local_size".to_string()
        ))
    }
    Ok(send_counts)
}

fn stable_split<T, F>(
    tsl: &mut [T],
    splitters: &[T],
    compare: F,
    comm: &dyn Communicator,
) -> Result<Vec<usize>>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    if !all_of(splitters.len() == comm.size() as usize - 1, comm) {
        bail!(Error::SplitterSizeError("Splitter not p - 1".to_string()));
    }
    // Locally find splitter positions in data
    //  (if an identical splitter appears at least three times (or more),
    //   then split the intermediary buckets evenly) => send_counts
    let mut send_counts: Vec<usize> = vec![0; comm.size() as usize];
    let mut i: usize = 0;
    let mut pos: usize = 0;

    while i < splitters.len() {
        // get the number of splitters which are equal starting from `i`
        let mut split_by: usize = 1;
        while i + split_by < splitters.len()
            && !compare(&splitters[i], &splitters[i + split_by]).is_lt()
        {
            split_by += 1;
        }
        // get the range of equal elements
        let eqr = equal_range_by(tsl, pos, &splitters[i], &compare);
        // assign smaller elements to processor left of splitter (= `i`)
        send_counts[i] += eqr.first - pos;
        pos = eqr.first;
        // split equal elements fairly across processors
        let eq_size = eqr.second - pos;
        // send equal elements to processor based on my own rank compared to
        // how many equal splitters there are
        if split_by == 1 {
            // Case 1) if there is only one splitter,
            //         assign equal elements to next processor (no splitting)
            send_counts[i + 1] += eq_size;
        } else {
            // Case 2) if there is >= 2 equal splitters:
            //         split processors into `split_by` regions
            let mut targetp =
                (comm.rank() as usize * split_by) / comm.size() as usize;
            if targetp >= split_by {
                targetp = split_by - 1
            };
            send_counts[i + 1 + targetp] += eq_size;
        }
        i += split_by;
        pos = eqr.second;
    }

    // send last elements to last processor
    let out_size = tsl.len() - pos;
    send_counts[comm.size() as usize - 1] += out_size;
    // variant check
    if !all_of(send_counts.iter().sum::<usize>() == tsl.len(), comm) {
        bail!(Error::SortInvariantError(
            "send_counts.iter().sum() === local_size".to_string()
        ))
    }
    Ok(send_counts)
}

pub fn samplesort<T, F>(
    tsl: &mut [T],
    compare: F,
    stable: bool,
    comm: &dyn Communicator,
) -> Result<()>
where
    T: Equivalence + Default + Clone,
    F: Fn(&T, &T) -> Ordering,
{
    // sample sort main steps
    // 1. local sort
    // 2. pick `s` samples regularly spaced on each processor
    // 3. bitonic sort samples
    // 4. allgather the last sample of each process -> splitters
    // 5. locally find splitter positions in data
    //    (if an identical splitter appears twice, then split evenly)
    //    => send_counts
    // 6. distribute send_counts with all2all to get recv_counts
    // 7. allocate enough space (may be more than previously allocated) for receiving
    // 8. all2allv
    // 9. local reordering (multiway-merge or again std::sort)
    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by sending elements to neighbors

    // p, rank, no, of samples (p - 1)
    let (p, rank, s) = (comm.size(), comm.rank(), comm.size() as usize - 1);

    // TODO:: timer start;
    if stable {
        tsl.sort_by(&compare);
    } else {
        tsl.sort_unstable_by(&compare);
    }

    // sequential case: we're done
    if p == 1 {
        return Ok(());
    }

    // local size & global sizes
    let local_size: usize = tsl.len();
    let global_size: usize = allreduce_sum(&local_size, comm);

    // check if we have a perfect block decomposition
    let mypart: ModuloDist = ModuloDist::new(global_size, p, rank);
    let _is_block_decomp: bool = all_of(local_size == mypart.local_size(), comm);

    // get splitters, using the method depending on whether the input consists
    // of arbitrary decompositions or not
    let local_splitters: Vec<T> = if _is_block_decomp {
        sample_block_decomp(tsl, &compare, s, comm)?
    } else {
        sample_arbit_decomp(tsl, &compare, s, comm)?
    };

    // 5. locally find splitter positions in data
    //    (if an identical splitter appears at least three times (or more),
    //    then split the intermediary buckets evenly) => send_counts
    let send_counts = if stable {
        stable_split(tsl, &local_splitters, &compare, comm)?
    } else {
        split(tsl, &local_splitters, &compare, comm)?
    };

    // MXX_ASSERT(!_AssumeBlockDecomp || (local_size <= (size_t)p || recv_n <= 2* local_size));
    let recv_counts = all2all_vec(&send_counts, comm)?;
    let recv_n: usize = recv_counts.iter().sum();
    if !all_of(
        !_is_block_decomp
            || (local_size <= p as usize)
            || recv_n <= 2 * local_size,
        comm,
    ) {
        bail!(Error::SortInvariantError(
            "!_is_block_decomp || (local_size <= p as usize) || recv_n <= 2 * local_size".to_string()
        ))
    }
    // TODO: use collective with iterators [begin,end) instead of pointers!
    let mut recv_elts = all2allv_vec(tsl, &send_counts, &recv_counts, comm)?;

    // 9. local reordering
    // TODO::: multi-way merge instead of sort
    if stable {
        recv_elts.sort_by(&compare);
    } else {
        recv_elts.sort_unstable_by(&compare);
    }

    // A. equalizing distribution into original size (e.g.,block decomposition)
    //    by elements to neighbors
    //    and save elements into the original iterator positions
    if _is_block_decomp {
        stable_distribute(&recv_elts, tsl, &mypart, comm)?;
    } else {
        arbit_distribute(&recv_elts, tsl, local_size, comm)?;
    }
    Ok(())
}
