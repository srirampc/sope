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

///
/// Light wapper around the SimpleCommunicator representing the default
/// MPI_COMM_WORLD with process '0' as the root process
pub struct WorldComm {
    _universe: Option<Universe>,
    pub comm: SimpleCommunicator,
    pub rank: i32,
    pub size: i32,
}

impl WorldComm {
    /// Calls MPI_Init
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

    /// Calls MPI_Finalize
    pub fn finalize(&self) -> i32 {
        unsafe { mpi::ffi::MPI_Finalize() }
    }

    /// Returns true if process id is 0, false otherwise
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }
}
