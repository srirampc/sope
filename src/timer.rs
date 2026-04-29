//! MPI-aware timing utilities.
//!
//! This module provides three timer types used throughout the library:
//!
//! * [`Timer`] - lightweight, process-local stopwatch.
//! * [`SectionTimer`] - barrier-synchronised timer that reduces a
//!   section's elapsed time across all processes (max / min / avg) and
//!   can log it as `INFO`.
//! * [`CumulativeTimer`] - accumulates several measured sections and
//!   reports the aggregate timings across all processes.

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
    collective::SystemOperation,
    traits::{Communicator, CommunicatorCollectives},
};
use std::{
    cell::RefCell, ops::AddAssign, time::{Duration, Instant}
};

use crate::{
    cond_info,
    reduction::{allreduce, allreduce_sum},
};

/// Time unit used by the timers when reporting elapsed durations.
pub enum TimerUnit {
    /// Whole seconds.
    Seconds,
    /// Milliseconds (10^-3 s).
    MilliSeconds,
    /// Microseconds (10^-6 s).
    MicroSeconds,
    /// Nanoseconds (10^-9 s).
    NanoSeconds,
}

/// Conversion of a [`std::time::Duration`] into a numeric value of a
/// chosen [`TimerUnit`].
pub trait UnitConverter {
    /// Convert `self` into a `u128` representation expressed in `units`.
    fn convert_to(&self, units: &TimerUnit) -> u128;
}

impl UnitConverter for Duration {
    fn convert_to(&self, units: &TimerUnit) -> u128 {
        match units {
            TimerUnit::MilliSeconds => self.as_millis(),
            TimerUnit::MicroSeconds => self.as_micros(),
            TimerUnit::NanoSeconds => self.as_micros(),
            TimerUnit::Seconds => self.as_secs() as u128,
        }
    }
}

/// Process-local stopwatch.
///
/// # Description
/// A simple timer that records its creation/reset time and reports the
/// elapsed time on demand in the configured [`TimerUnit`]. Unlike
/// [`SectionTimer`], it does not perform any MPI synchronisation.
///
/// # Examples
/// \```
/// let mut t = Timer::new_millis();
/// // ... do some work ...
/// let ms = t.elapsed();
/// t.reset();
/// \```
pub struct Timer {
    start: Instant,
    units: TimerUnit,
}

impl Timer {
    /// Create a new local timer measuring in the given `units`.
    pub fn new(units: TimerUnit) -> Self {
        Self {
            start: Instant::now(),
            units,
        }
    }

    /// Convenience constructor for a millisecond-precision timer.
    pub fn new_millis() -> Self {
        Self::new(TimerUnit::MilliSeconds)
    }

    /// Return the elapsed time since creation/last reset, in `units`.
    pub fn elapsed(&self) -> u128 {
        self.start.elapsed().convert_to(&self.units)
    }

    /// Restart the timer.
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

/// Barrier-synchronised section timer.
///
/// # Description
/// `SectionTimer` is intended to time a code section that runs across
/// several MPI processes. On construction it issues an
/// [`mpi::traits::CommunicatorCollectives::barrier`] so that every rank
/// starts measuring at the same moment. Calls to [`SectionTimer::end_section`]
/// or [`SectionTimer::info_section`] reduce the per-rank elapsed times
/// into the maximum, minimum and average across the communicator.
pub struct SectionTimer<'a> {
    comm: &'a dyn Communicator,
    root: i32,
    sep: String,
    units: TimerUnit,
    start: RefCell<Instant>,
}

impl<'a> SectionTimer<'a> {
    /// Construct a new section timer.
    ///
    /// # Arguments
    /// * `comm` - Communicator across which the section is timed.
    /// * `root` - rank that emits the `INFO` log line.
    /// * `units` - units for elapsed times.
    /// * `sep` - separator used in the log output.
    ///
    /// # Notes
    /// A barrier is issued so that every rank starts timing together.
    pub fn new(
        comm: &'a dyn Communicator,
        root: i32,
        units: TimerUnit,
        sep: &str,
    ) -> Self {
        comm.barrier();
        Self {
            comm,
            root,
            units,
            sep: sep.to_string(),
            start: RefCell::new(Instant::now()),
        }
    }

    /// Convenience constructor with rank `0` as root and millisecond
    /// precision.
    pub fn from_comm(comm: &'a dyn Communicator, sep: &str) -> Self {
        Self::new(comm, 0, TimerUnit::MilliSeconds, sep)
    }

    /// Reduce a per-rank section time into `(max, min, avg)` across the
    /// communicator. A barrier is issued at the end so that subsequent
    /// timing starts in lockstep across all ranks.
    pub(super) fn reduce_section_time(
        &self,
        sec_time: f64,
    ) -> (f64, f64, f64) {
        let max_time = allreduce(&sec_time, self.comm, SystemOperation::max());
        let min_time = allreduce(&sec_time, self.comm, SystemOperation::min());
        let sum_time = allreduce_sum(&sec_time, self.comm);
        let avg_time = sum_time / self.comm.size() as f64;
        self.comm.barrier();
        (max_time, min_time, avg_time)
    }

    /// Local elapsed time since creation/last reset, in the configured
    /// units, as `f64`.
    pub fn elapsed(&self) -> f64 {
        self.start.borrow().elapsed().convert_to(&self.units) as f64
    }

    /// End the current section and return `(max, min, avg)` of the
    /// per-rank elapsed times across the communicator.
    pub fn end_section(&self) -> (f64, f64, f64) {
        let sec_time = self.elapsed();
        self.reduce_section_time(sec_time)
    }

    /// Restart the timer.
    pub fn reset(&self) {
        self.start.replace(Instant::now());
    }

    /// End the section and emit an `INFO` log line containing
    /// `max`, `min`, `avg` elapsed times and the section `name`.
    ///
    /// The line is only emitted by the `root` rank. The timer is
    /// reset at the end so it can be reused for the next section.
    pub fn info_section(&self, name: &str) {
        if log::log_enabled!(log::Level::Info) {
            let (max_time, min_time, avg_time) = self.end_section();
            self.reset();
            cond_info!(
                self.comm.rank() == self.root;
                "TIMER{}{:.3}{}{:.3}{}{:.3}{}{} ",
                self.sep, max_time,
                self.sep, min_time,
                self.sep, avg_time,
                self.sep, name
            )
        }
    }
}

/// Cumulative MPI section timer.
///
/// # Description
/// Wraps a [`SectionTimer`] and adds an accumulator. Use
/// [`CumulativeTimer::reset`] before each measured section,
/// [`CumulativeTimer::add_elapsed`] after each section to add its
/// elapsed time into the accumulator, and finally
/// [`CumulativeTimer::info_region`] to log the reduced
/// (`max`, `min`, `avg`) of the cumulated time across the communicator.
pub struct CumulativeTimer<'a> {
    s_timer: SectionTimer<'a>,
    total_elapsed: RefCell<f64>,
}

impl<'a> CumulativeTimer<'a> {
    /// Construct a new cumulative timer.
    ///
    /// # Arguments
    /// * `comm` - Communicator across which sections are timed.
    /// * `root` - rank that emits the `INFO` log line.
    /// * `units` - units for elapsed times.
    /// * `sep` - separator used in the log output.
    ///
    /// A barrier is issued so that every rank starts in lockstep.
    pub fn new(
        comm: &'a dyn Communicator,
        root: i32,
        units: TimerUnit,
        sep: &str,
    ) -> Self {
        comm.barrier();
        Self {
            s_timer: SectionTimer::new(comm, root, units, sep),
            total_elapsed: RefCell::new(0.0),
        }
    }

    /// Convenience constructor with rank `0` as root and millisecond
    /// precision.
    pub fn from_comm(comm: &'a dyn Communicator, sep: &str) -> Self {
        Self::new(comm, 0, TimerUnit::MilliSeconds, sep)
    }

    /// Restart the underlying section timer (call before each timed
    /// region).
    pub fn reset(&self) {
        self.s_timer.reset();
    }

    /// Add the elapsed time since the last [`reset`](Self::reset) to
    /// the cumulative total.
    pub fn add_elapsed(&self) {
        let elapsed = self.s_timer.elapsed();
        self.total_elapsed.borrow_mut().add_assign(elapsed);
    }

    /// Emit an `INFO` log line for the cumulated region containing
    /// `max`, `min`, `avg` of the total elapsed time across the
    /// communicator and the region `name`.
    ///
    /// The line is only emitted by the `root` rank.
    pub fn info_region(&self, name: &str) {
        if log::log_enabled!(log::Level::Info) {
            let rv = *self.total_elapsed.borrow();
            let (max_time, min_time, avg_time) =
                self.s_timer.reduce_section_time(rv);
            cond_info!(
                self.s_timer.comm.rank() == self.s_timer.root;
                "CR TIMER{}{:.3}{}{:.3}{}{:.3}{}{} ",
                self.s_timer.sep, max_time,
                self.s_timer.sep, min_time,
                self.s_timer.sep, avg_time,
                self.s_timer.sep, name
            )
        }
    }
}
