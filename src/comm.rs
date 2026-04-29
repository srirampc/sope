//! Lightweight wrappers around the MPI environment / world communicator.
//!
//! The [`WorldComm`] type owns the optional [`mpi::environment::Universe`]
//! handle returned by [`mpi::initialize`] and exposes the resulting
//! [`mpi::topology::SimpleCommunicator`] (`MPI_COMM_WORLD`) along with
//! the local rank and total size for convenience. Dropping a `WorldComm`
//! that owns the `Universe` will finalise MPI automatically; an explicit
//! [`WorldComm::finalize`] is also provided for cases where the
//! application wants to call `MPI_Finalize` directly.

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

use mpi::{
    environment::Universe, topology::SimpleCommunicator, traits::Communicator,
};

/// Light wrapper around `MPI_COMM_WORLD`.
///
/// # Description
/// Holds the (optional) [`Universe`] returned by `mpi::initialize` and
/// caches the world `comm`, the local `rank`, and the world `size` so
/// that callers do not have to repeatedly query them. Process `0` is
/// treated as the conventional root.
///
/// # Examples
/// \```
/// let c = WorldComm::init();
/// if c.is_root() {
///     println!("World size = {}", c.size);
/// }
/// \```
pub struct WorldComm {
    /// Owned `Universe` handle. `None` when MPI was already initialised
    /// by another part of the application before [`WorldComm::init`]
    /// was called.
    _universe: Option<Universe>,
    /// Initialised [`mpi::topology::SimpleCommunicator`] for
    /// `MPI_COMM_WORLD`.
    pub comm: SimpleCommunicator,
    /// Rank of the current process inside `MPI_COMM_WORLD`.
    pub rank: i32,
    /// Total number of processes in `MPI_COMM_WORLD`.
    pub size: i32,
}

impl WorldComm {
    /// Initialise MPI and capture `MPI_COMM_WORLD`.
    ///
    /// # Description
    /// Calls [`mpi::initialize`] (i.e. `MPI_Init`) on first invocation
    /// and stores the returned [`Universe`] handle so that MPI is
    /// automatically finalised when this `WorldComm` is dropped. If
    /// MPI has already been initialised elsewhere, falls back to
    /// `SimpleCommunicator::world()` and stores `None` for the
    /// universe (so this instance will not finalise MPI on drop).
    ///
    /// # Returns
    /// A fully populated `WorldComm` whose `rank` and `size` are
    /// already cached.
    pub fn init() -> Self {
        let (comm, _universe) = match mpi::initialize() {
            Some(universe) => (universe.world(), Some(universe)), // First time init
            None => (SimpleCommunicator::world(), None), // Already initialized
        };
        WorldComm {
            rank: comm.rank(),
            size: comm.size(),
            _universe,
            comm,
        }
    }

    /// Explicitly call `MPI_Finalize`.
    ///
    /// # Description
    /// Wraps the unsafe FFI call [`mpi::ffi::MPI_Finalize`]. Most
    /// applications can rely on the automatic finalisation that
    /// happens when the owning [`Universe`] (held inside this
    /// `WorldComm`) is dropped; use this when you need to control the
    /// timing of `MPI_Finalize` explicitly.
    ///
    /// # Returns
    /// The `i32` status code returned by `MPI_Finalize`.
    pub fn finalize(&self) -> i32 {
        unsafe { mpi::ffi::MPI_Finalize() }
    }

    /// Returns `true` if this process is the root (rank `0`).
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }
}
