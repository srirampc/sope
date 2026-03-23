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

pub enum TimerUnit {
    Seconds,
    MilliSeconds,
    MicroSeconds,
    NanoSeconds,
}

pub trait UnitConverter {
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

pub struct Timer {
    start: Instant,
    units: TimerUnit,
}

impl Timer {
    pub fn new(units: TimerUnit) -> Self {
        Self {
            start: Instant::now(),
            units,
        }
    }

    pub fn new_millis() -> Self {
        Self::new(TimerUnit::MilliSeconds)
    }

    pub fn elapsed(&self) -> u128 {
        self.start.elapsed().convert_to(&self.units)
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

pub struct SectionTimer<'a> {
    comm: &'a dyn Communicator,
    root: i32,
    sep: String,
    units: TimerUnit,
    start: RefCell<Instant>,
}

impl<'a> SectionTimer<'a> {
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

    pub fn from_comm(comm: &'a dyn Communicator, sep: &str) -> Self {
        Self::new(comm, 0, TimerUnit::MilliSeconds, sep)
    }

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

    pub fn elapsed(&self) -> f64 {
        self.start.borrow().elapsed().convert_to(&self.units) as f64
    }

    pub fn end_section(&self) -> (f64, f64, f64) {
        let sec_time = self.elapsed();
        self.reduce_section_time(sec_time)
    }

    pub fn reset(&self) {
        self.start.replace(Instant::now());
    }

    // ends section and
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

pub struct CumulativeTimer<'a> {
    s_timer: SectionTimer<'a>,
    total_elapsed: RefCell<f64>,
}

impl<'a> CumulativeTimer<'a> {
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

    pub fn from_comm(comm: &'a dyn Communicator, sep: &str) -> Self {
        Self::new(comm, 0, TimerUnit::MilliSeconds, sep)
    }

    pub fn reset(&self) {
        self.s_timer.reset();
    }

    pub fn add_elapsed(&self) {
        let elapsed = self.s_timer.elapsed();
        self.total_elapsed.borrow_mut().add_assign(elapsed);
    }

    // ends section and
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
