# sope derive

Derive macros to support generic types for MPI operations.

### Generic MPI types

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

## Licensing

Our code is licensed under the
**Apache License 2.0** (see [`LICENSE`](LICENSE)).
