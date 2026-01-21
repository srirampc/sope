# sope

`sope` is a Rust port of [mxx](https://github.com/patflick/mxx),
a `C++` library for `MPI` .
Similar to `mxx`, the main goal of this library is to provide two things:

1. Simplified, and type-safe bindings to common MPI operations with error handling.
2. A collection of scalable, high-performance standard algorithms for parallel
   distributed memory architectures, such as sorting.

As such, `sope` is useful for rapid `MPI` algorithm development, prototyping,
and deployment.

### Features

- Built on top of `rsmpi` MPI library for rust and all functions are
  type-genric w.r.t the `rsmpi`'s `Equivalence` trait.
- Collective operations with input/output size validations, and error handling.
- Convenience functions and overloads for common MPI operations with
  sane defaults (e.g., super easy collectives: `let allsizes: Vec<usize> =
sope::reduction::allgather_one(local_size, &comm)`).
- Parallel sorting with similar to standard `sort` (`sope::sort::sort`)

### Planned / TODO

- [ ] Simplify user operations
- [ ] Send/Receive operations
- [ ] Wrappers for non-blocking collectives
- [ ] Implementing and tuning different sorting algorithms
- [ ] More parallel (standard) algorithms
- [ ] Serialization/de-serialization of non contiguous data types (maybe)
- [ ] Parallel random number engines (for use with `rand` library implementations)
- [ ] Communicator classes for different topologies
- [ ] Full-code and intro documentations
- [ ] Increase test coverage.

### Status

Currently `sope` is a small personal project at very early stages.

### Examples

The folder `examples` folder contains usage examples for reductions, collective
operations, balanced block distributions, and parallel sorting.

#### Collective Operations

This example shows the main features of `sope`'s wrapper for MPI collective
operations:

- datatype deduction based on rsmpi's `Equivalence` trait
- convenience functions for `Vec`, both for collective operations

```rust
   use sope::collective::allgatherv_full_vec;

   let universe = mpi::initialize().unwrap();
   let comm = universe.world();
   // local numbers, can be different size (multiple of p) on each process
   let local_numbers: Vec<usize> = ...;
   // allgather the local numbers, easy as pie:
   let all_numbers: Vec<usize> = allgatherv_full_vec(local_numbers, &comm);
```

#### Reductions

The following example showcases the interface to reductions:

```rust
   use mpi::collective::UserOperation;
   use mpi::{topology::Communicator, traits::Equivalence};
   use sope::reduction::all_reduce;

   #[derive(Debug, Equivalence, Default, Clone)]
   struct CPair {
       first: f32,
       second: i32,
   }

   let universe = mpi::initialize().unwrap();
   let comm = universe.world();

   // let v be distributed pairs and
   let v: CPair = ...;
   // find the one with the max second element
   let max_op = UserOperation::commutative(|x, y| {
        let x: &[CPair] = x.downcast().unwrap();
        let y: &mut [CPair] = y.downcast().unwrap();
        if x[0].second > y[0].second {
            y[0].first = x[0].first;
            y[0].second = x[0].second.clone();
        }
    });
   let max_pair = all_reduce(&v, &c.comm, max_op);
```

#### Sorting

Consider a simple example, where you might want to sort tuples `(key: i32, 
x: f32,  y: f32)` by key `key` in parallel using `MPI`. Doing so in pure C/MPI
requires quite a lot of coding (~100 lines), debugging, and frustration. Thanks
to `sope` and `rsmpi`, this becomes as easy as:

```rust
   #[derive(Debug, Equivalence, Default, Clone)]
   struct CPair {
       key: i32,
       x: f32,
       y: f32,
   }

   // define a comparator for the tuple
   let cmp = |a: &CPair, b: &CPair| a.key.cmp(&b.key);

    // fill the vector in each process ...
   let mut data: Vec<CPair> = ..

    // call sope::sort to do all the heavy lifting:
   sope::sort::sort_by(&mut v, cmp, &comm)?;
```

In the background, `sope` performs many things, including (but not limited to):

- distributing the data if not yet done so
- calling standard `sort` as a local base case, in case the communicator consists of a
  single processor, `sope::sort` will fall-back to `std::sort`
- redistributing the data so that it has the same distribution as given in the
  input to `sort`

### Authors

- Patrick Flick (Original author of [mxx](https:://github.com/patflick/mxx)),
  from which many of the implementations are ported
- Sriram Chockalingam

## Installation

Add this repository as a dependency in Cargo.toml.

### Dependencies

`sope` currently works with `MPI-2` and `MPI-3`.
However, some collective operations and sorting will work on data sizes `>= 2 GB` only with `MPI-3`.

### Compiling

Compile with cargo, similar to any other Rust library.

```sh
cargo build
```

#### Building tests

All the tests are in the example directory and can be compiled using `cargo`:

```sh
cargo b --examples
```

Running the tests (with however many processes you want).

```sh
mpirun -np 13 ./target/debug/$SOPE_EXAMPLE_EXECUTABLE
```

## Licensing

Our code is licensed under the
**Apache License 2.0** (see [`LICENSE`](LICENSE)).
