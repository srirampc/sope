# AGENTS.md

Operational guide for AI agents (and future contributors) working on the
**sope** crate. Keep this file up to date when you change architecture,
public APIs, or build/test workflows.

## What is sope?

`sope` is a Rust port of the C++ template library
[`mxx`](https://github.com/patflick/mxx). It builds on top of
[`rsmpi`](https://docs.rs/mpi) and provides:

1. Simplified, type-safe wrappers around common MPI operations with input
   validation and `anyhow::Result`-based error handling.
2. A collection of high-performance distributed algorithms (sort,
   redistribution, reductions, …).

## Repository layout

```
sope/
├── Cargo.toml              # workspace root; member: . and sope-derive
├── sope-derive/            # proc-macro crate (GEquivalence derive)
├── src/                    # library source (see “Architecture”)
├── examples/               # example binaries + run script
└── target/                 # build output
```

## Architecture / internal APIs

The library is organised as one top-level module per concern. All modules
live under `src/` and are re-exported via `src/lib.rs`.

| Module             | Purpose                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `comm`             | `WorldComm` wrapper around `MPI_COMM_WORLD` (init / finalize / `is_root`).                    |
| `bcast`            | `bcast_one`, `bcast`, `bcast_vec`: typed broadcast helpers.                                   |
| `shift`            | Left/right shift of one element or a vector along the rank axis.                              |
| `reduction`        | `reduce`, `allreduce`, `scan`, `exclusive_scan`, predicate reductions (`all_of`, `any_of`, `none_of`, `all_same`), and `min/max_element*` with rank tracking. |
| `collective`       | Slice/Vec wrappers around `scatter`, `scatterv`, `gather`, `gatherv`, `allgather`, `allgatherv`, `all2all`, `all2allv`. Auto-falls back to `collective::big` when any per-rank count exceeds `i32::MAX`. |
| `collective::big`  | Large-message variants implemented via non-blocking point-to-point (`immediate_send`/`receive`), plus slower scatter-by-scatter variants. |
| `partition`        | `Dist` trait + concrete distributions: `ModuloDist`, `InterleavedDist`, `ArbitDist`.          |
| `distribution`     | Re-distribution strategies that implement the `Distributor<T>` trait: `Over2UnderDistributor`, `SurplusDistributor`, `StableDistributor`, `ArbitDistributor`. Convenience free functions: `distribute_scatter`, `stable_distribute*`, `distribute_vec`, `arbit_distribute`. |
| `sort`             | `bitonic_sort`, `samplesort`, plus `sort` / `stable_sort` (and `_by` variants) and distributed `is_sorted*` predicates. |
| `timer`            | `Timer`, `SectionTimer`, `CumulativeTimer` for MPI-aware timing.                              |
| `log`              | Macros for conditional logging (`cond_info!`, …), gathered logging (`gather_info!`, …), and `anyhow`-based `ensure!` / `ensure_eq!` wrappers. |
| `util`             | Generic helpers: `Pair<T1, T2>`, `inc_prefix_sum`, `exc_prefix_sum*`, `which_itr`, `equal_range*`. |
| `traits`           | Re-exports `sope_derive::GEquivalence` (derive macro for MPI `Equivalence`). |

Crate-level items in `src/lib.rs`:

- `MCount` blanket trait: integer count types usable with collectives.
- `All2allvArgs<T>` struct holding send/recv counts and displacements with
  `to_i32()` / `to_usize()` conversions and `from_counts(...)` builder.

### Conventions

- Public functions return `anyhow::Result<...>`. Errors are typed per
  module (e.g. `collective::Error`, `sort::Error`) and converted into
  `anyhow::Error` via `bail!`.
- For each collective there are typically three flavours:
  - in-place into a caller-supplied slice (`scatter`, `gatherv`, …),
  - allocating variant returning a `Vec<T>` (`*_vec` suffix),
  - validation helpers (`validate_*`) that run the same collective
    pre-conditions used by the wrappers.
- All collectives validate sizes across ranks using
  `reduction::all_of`, `any_of`, `all_same` before issuing the underlying
  `rsmpi` call.

## Build / test commands

The crate uses Rust 2024 edition. A working MPI installation
(`mpicc` / `mpirun` in `PATH`) is required because `rsmpi` links against
the system MPI.

```bash
# Compile the library
cargo build --lib

# Compile everything, including the examples
cargo build --examples

# Run unit tests (no MPI required for the few non-MPI tests)
cargo test

# Lints / formatting
cargo clippy --all-targets
cargo fmt --all
```

### Running the examples as integration tests

The script `examples/run_sope_examples.sh` runs every `examples/sope_*.rs`
binary under `mpirun -np 4` with `RUST_LOG=info`. It is the canonical way
to validate end-to-end behaviour:

```bash
cargo build --examples
./examples/run_sope_examples.sh
```

You can use this script as the project's de-facto integration test suite —
it exercises broadcast, collective, derive, distribution, reductions,
shift and sort entry points across 4 MPI processes.

## Documentation

Module-level overviews and per-function docs follow the style established
in `bcast.rs` / `shift.rs`:

- `# Description`, `# Arguments`, `# Returns`, `# Errors`, `# Examples`
  sections where appropriate.
- License header (Apache-2.0, Georgia Institute of Technology) at the top
  of every source file.

Generate the rendered HTML docs with:

```bash
cargo doc --no-deps --open
```

## Tips for agents

- Never edit files inside `target/`.
- When adding a new collective wrapper, mirror the existing pattern:
  validation helper → core slice variant → `*_vec` allocating wrapper, and
  add a corresponding example under `examples/`.
- After non-trivial changes, run `cargo build --lib` and, if MPI is
  available, `./examples/run_sope_examples.sh` to validate behaviour at
  4 ranks.
- If a new `Distributor`, `Dist` or sort algorithm is added, list it in
  the architecture table above so future readers can find it.
