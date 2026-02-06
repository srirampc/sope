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
- `GEquivalence` derive macro for automatically constructing MPI_Datatypes
  for rust generic types.
- Collective operations with input/output size validations, and error handling.
- Convenience functions and overloads for common MPI operations with
  sane defaults (e.g., super easy collectives: `let allsizes: Vec<usize> =
sope::reduction::allgather_one(local_size, &comm)`).
- Parallel sorting with similar to standard `sort` (`sope::sort::sort`)

### Planned / TODO

- [ ] Send/Receive operations
- [ ] Simplify user operations
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


#### Generic MPI types

`sope::traits::GEquivalence` is a derive macro that can automatically define 
MPI Datatypes for a struct with generics. 

The following example defines struct `GenericTriple`, which has three members
of three different genric types. By deriving from the `GEquivalence` macro,
we can use elements of this type as a MPI datatype for distributed sort.

```rust
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::iter::zip;
use sope::{
    comm::WorldComm,
    sort::{sort, is_sorted_by},
    traits::GEquivalence,
};

#[derive(GEquivalence, Debug, Clone, Default)]
struct GenericTriple<T1, T2, T3> {
    first: T1,
    second: T2,
    third: T3,
}

type TTuple = GenericTriple<i32, u16, f32>;

let c = WorldComm::init();
let irng = ChaCha8Rng::seed_from_u64(0).random_iter::<i32>();
let urng = ChaCha8Rng::seed_from_u64(0).random_iter::<u16>();
let frng = ChaCha8Rng::seed_from_u64(0).random_iter::<f32>();

let mut tvec: Vec<TTuple> = std::iter::zip(irng, urng)
    .zip(frng)
    .take(nelts)
    .map(|((x, y), z)| TTuple {
        first: x,
        second: y,
        third: z,
    })
    .collect();

let cmp = |a: &TTuple, b: &TTuple| (a.first, a.second).cmp(&(b.first, b.second));
sort_by(&mut tvec, cmp, &c.comm)?;

let d_sorted = is_sorted_by(&lvec,|a, b| cmp_fn(a, b).is_le(), &c.comm)?;
assert!(d_sorted)
```

### Authors

- Patrick Flick (Original author of [mxx](https:://github.com/patflick/mxx)),
  from which many of the implementations are ported
- Sriram Chockalingam

## Installation

Add this repository as a dependency in Cargo.toml.

### Dependencies

`sope` currently works with `MPI-2` and `MPI-3`. However, some collective 
operations and sorting will work on data sizes `>= 2 GB` only with `MPI-3`.

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

## Library Name

The library is named 'sope' after the [Sope Creek](https://en.wikipedia.org/wiki/Sope_Creek),
a " 11.6-mile-long (18.7 km) stream located in Cobb County, Georgia, United States."
The creek's name itself has an interesting 
[etymology](https://en.wikipedia.org/wiki/Sope_Creek#False_etymology) on its own. 
As per wikipedia, a 2012 statement in the  __Atlanta Journal__ states:

    It was originally called Soap Creek and was named after a Cherokee named Soap or
    "Old Soap," according to Jeff Bishop, president of the Georgia chapter of the 
    Trail of Tears Association, which works with the Cherokee Nation, the Eastern
    Band of Cherokee Indians and the National Park Service to preserve Trail of
    Tears related sites in Georgia. Bishop wrote in an email that it was called
    Soap Creek on the 1832 Georgia Land Lottery maps. Old Soap was highly regarded
    by the whites in the area, according to "The First Hundred Years: A Short 
    History of Cobb County," which states that "he had lived there so long that
    a creek and its branch were named for him." However, there was a dispute,
    and he and his family were forced to move to Cherokee County, where they lived 
    until they were relocated on the Trail of Tears, wrote Bishop, who contributes
    to www.trailofthetrail.com. "There are descendants of Soap who now live in 
    Oklahoma, in the Cherokee Nation," Bishop wrote in an email. "Chris Soap 
    serves on the Cherokee Nation tribal council, and his father, Charley Soap,
    is a respected elder who is the widower of former Cherokee Nation Chief Wilma
    Mankiller." The spelling of Sope Creek apparently was changed sometime in the
    19th century, but it was still spelled Soap in 1849's "Statistics of the 
    State of Georgia." Historical markers use both Soap and Sope.

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5a/Sope_Creek_December_2019.jpg" alt="Sope Creek" width="350" height="233"/>
<figcaption>
Sope Creek (Image taken from Wikipedia)
</figcaption>
</figure>

## Licensing

Our code is licensed under the
**Apache License 2.0** (see [`LICENSE`](LICENSE)).
