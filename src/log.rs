//! Logging utilities for parallel programs.
//!
//! This module provides three families of macros:
//!
//! * **Conditional logging** (`cond_println!`, `cond_eprintln!`,
//!   `cond_info!`, `cond_error!`, `cond_debug!`, `cond_warn!`) -
//!   emit a message only when a user-supplied boolean expression is
//!   `true`. Typically used to restrict output to the root process,
//!   e.g. `cond_info!(comm.rank() == 0; "...")`.
//!
//! * **Gathered logging** (`gather_format!`, `gather_format_vec!`,
//!   `gather_format_tstamp_vec!`, `log_gather_format_vec!`,
//!   `gather_println!`, `gather_eprintln!`, `gather_info!`,
//!   `gather_error!`, `gather_debug!`, `gather_warn!`) - format a
//!   per-rank message, prefix it with the rank (and optionally a
//!   timestamp), gather all per-rank strings on rank `0` via
//!   [`crate::collective::gather_strings`] and print/log the result
//!   in rank order. Empty messages are dropped.
//!
//! * **`anyhow`-based ensures** (`ensure!`, `ensure_eq!`) - thin
//!   wrappers around `anyhow::ensure!` that attach the source file
//!   and line number to the [`EnsureError`] payload, and for
//!   `ensure_eq!` also embed the left/right values for easier
//!   debugging.

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

use std::fmt::{Debug, Display};
use thiserror::Error;

/// Error payload produced by the [`crate::ensure!`] / [`crate::ensure_eq!`] macros.
///
/// # Description
/// `EnsureError` records the source file (`F`), line (`T`) and -
/// for `ensure_eq!` - the offending left (`L`) and right (`R`)
/// values. The variant `C` is used by the unary `ensure!` macro and
/// only carries the location; the variant `LR` is produced by
/// `ensure_eq!` and includes both values so that they appear in the
/// resulting error message.
#[derive(Error, Debug)]
pub enum EnsureError<F: Debug + Display, T: Debug + Display, L: Debug, R: Debug> {
    /// Plain "condition was false" failure raised by [`crate::ensure!`].
    #[error("ensure failed at {0}:{1}")]
    C(F, T, L, R),
    /// "Left != Right" failure raised by [`crate::ensure_eq!`].
    #[error(
        "ensure `left == right` failed at {0}:{1} :: left:({2:?}), right:({3:?})"
    )]
    LR(F, T, L, R),
}

/// Conditionally call [`println!`].
///
/// # Description
/// Evaluates the condition expression first; the formatted message
/// is only printed when the condition is `true`. Useful to restrict
/// console output to a single rank.
///
/// # Examples
/// \```
/// cond_println!(comm.rank() == 0; "hello from root, n = {}", n);
/// \```
#[macro_export]
macro_rules! cond_println {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            println!($($args)*)
        }
    };
}

/// Conditionally call [`eprintln!`].
///
/// # Description
/// Same as [`cond_println!`] but writes to standard error.
#[macro_export]
macro_rules! cond_eprintln {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            eprintln!($($args)*)
        }
    };
}

/// Conditionally emit a `log::info!` message.
///
/// # Description
/// Equivalent to `if cond { log::info!(...) }` but additionally
/// short-circuits on `log::log_enabled!(Info)` so that the
/// condition expression is only evaluated when info logging is on.
///
/// # Examples
/// \```
/// cond_info!(comm.rank() == 0; "starting phase {}", name);
/// \```
#[macro_export]
macro_rules! cond_info {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(log::Level::Info) {
            if $cond_expr {
                ::log::info!($($args)*)
            }
        }
    };
}

/// Conditionally emit a `log::error!` message.
///
/// # Description
/// Same shape as [`cond_info!`] but at the `Error` level.
#[macro_export]
macro_rules! cond_error {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!( log::Level::Error) {
            if $cond_expr {
                ::log::error!($($args)*)
            }
        }
    };
}

/// Conditionally emit a `log::debug!` message.
///
/// # Description
/// Same shape as [`cond_info!`] but at the `Debug` level.
#[macro_export]
macro_rules! cond_debug {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(log::Level::Debug) {
            if $cond_expr {
                ::log::debug!($($args)*)
            }
        }
    };
}

/// Conditionally emit a `log::warn!` message.
///
/// # Description
/// Same shape as [`cond_info!`] but at the `Warn` level.
#[macro_export]
macro_rules! cond_warn {
    ($cond_expr: expr; $($args:tt)* ) => {
        if ::log::log_enabled!(log::Level::Warn) {
            if $cond_expr {
                ::log::warn!($($args)*)
            }
        }
    };
}

/// Format a per-rank message and gather all messages on rank 0.
///
/// # Description
/// Each rank formats its own message using the `format!` arguments.
/// Non-empty messages are prefixed with `[Rxxxx] ` (rank in a 4-wide
/// field) before being collected on rank `0` via
/// [`crate::collective::gather_strings`]. Returns an
/// `anyhow::Result<Option<Vec<String>>>`: `Some(vec)` on rank `0`,
/// `None` on every other rank.
///
/// # Examples
/// \```
/// let lines = gather_format_vec!(&comm; "value = {}", v)?;
/// if let Some(lines) = lines { for l in lines { println!("{l}"); } }
/// \```
#[macro_export]
macro_rules! gather_format_vec {
    ($comm_expr: expr; $($args:tt)* ) => {{
        let frs = format!($($args)*);
        let s = if !frs.is_empty() {
            use ::mpi::topology::Communicator;
            format!("[R{:>4}] {}", ($comm_expr).rank(), frs)
        } else {
            frs
        };
        $crate::collective::gather_strings(s, 0, ($comm_expr))
    }};
}

/// Same as [`gather_format_vec!`] but also embeds a local timestamp.
///
/// # Description
/// Per-rank messages are prefixed with `[Rxxxx::YYYY-MM-DD HH:MM:SS]`
/// using `chrono::Local::now()` and gathered on rank `0`. Empty
/// messages are dropped.
#[macro_export]
macro_rules! gather_format_tstamp_vec {
    ($comm_expr: expr; $($args:tt)* ) => {{
        let frs = format!($($args)*);
        let s = if !frs.is_empty() {
            use ::mpi::topology::Communicator;
            format!(
                "[R{:>4}::{}] {}", ($comm_expr).rank(),
                ::chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                frs
            )
        } else {
            frs
        };
        $crate::collective::gather_strings(s, 0, ($comm_expr))
    }};
}

/// Log-level guarded version of [`gather_format_vec!`].
///
/// # Description
/// Only performs the gather when the requested `log_level` is
/// enabled. On error, returns a single-string vector containing the
/// error message (so the caller can still log something useful).
/// Returns `Option<Vec<String>>`: `Some(...)` on rank `0` when
/// logging is enabled, `None` otherwise.
#[macro_export]
macro_rules! log_gather_format_vec {
    ($comm_expr: expr; $log_level: expr; $($args:tt)* ) => {
        if ::log::log_enabled!($log_level) {
            match $crate::gather_format_vec!($comm_expr; $($args)*) {
                ::anyhow::Result::Ok(g_in) => {
                    g_in
                }
                ::anyhow::Result::Err(err) => {
                    use ::mpi::traits::Communicator;
                    if ($comm_expr).rank() == 0 {
                        Some(vec![err.to_string()])
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        }
    };
}

/// Gather per-rank formatted messages, returning a `Vec<String>`.
///
/// # Description
/// Convenience wrapper around [`gather_format_vec!`] that flattens
/// the `Result<Option<Vec<...>>>` into a plain `Vec<String>`. On
/// non-root ranks the vector is empty; on errors it contains a
/// single string with the error message.
#[macro_export]
macro_rules! gather_format {
    ($comm_expr: expr; $($args:tt)* ) => {
        match $crate::gather_format_vec!($comm_expr; $($args)*) {
            ::anyhow::Result::Ok(rsv) => {
                rsv.unwrap_or(vec![])
            }
            ::anyhow::Result::Err(err) => {
                vec![err.to_string()]
            }
        }
    };
}

/// Gather per-rank messages and `println!` them on rank `0`.
///
/// # Description
/// On rank `0` the gathered messages are printed one per line in
/// rank order; other ranks print nothing.
#[macro_export]
macro_rules! gather_println {
    ($comm_expr: expr; $($args:tt)* ) => {
        for s in $crate::gather_format!($comm_expr; $($args)*) {
            println!("{}", s);
        }
    };
}

/// Gather per-rank messages and `eprintln!` them on rank `0`.
///
/// # Description
/// On rank `0` the gathered messages are printed on stderr one per line in
/// rank order; other ranks print nothing.
#[macro_export]
macro_rules! gather_eprintln {
    ($comm_expr: expr;$($args:tt)* ) => {
        for s in $crate::gather_format!($comm_expr; $($args)*) {
            eprintln!("{}", s);
        }
    };
}

/// Gather per-rank messages and emit them as `log::info!` on rank
/// `0`.
///
/// # Description
/// Short-circuits when `Info` logging is disabled. The gather is
/// guarded by [`log_gather_format_vec!`].
#[macro_export]
macro_rules! gather_info {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = $crate::log_gather_format_vec!(
            $comm_expr; ::log::Level::Info; $($args)*
        ) {
            for fs in fsv {
                ::log::info!("{}", fs);
            }
        }
    }
}

/// Gather per-rank messages and emit them as `log::error!` on rank `0`.
///
/// # Description
/// Short-circuits when `Error` logging is disabled. The gather is
/// guarded by [`log_gather_format_vec!`].
#[macro_export]
macro_rules! gather_error {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = $crate::log_gather_format_vec!(
            $comm_expr; log::Level::Error; $($args)*
        ) {
            for fs in fsv {
                log::error!("{}", fs);
            }
        }
    }
}

/// Gather per-rank messages and emit them as `log::debug!` on rank `0`.
///
/// # Description
/// Short-circuits when `Debug` logging is disabled. The gather is
/// guarded by [`log_gather_format_vec!`].
#[macro_export]
macro_rules! gather_debug {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = $crate::log_gather_format_vec!(
            $comm_expr; log::Level::Debug; $($args)*
        ) {
            for fs in fsv {
                log::debug!("{}", fs);
            }
        }
    }
}

/// Gather per-rank messages and emit them as `log::warn!` on rank `0`.
///
/// # Description
/// Short-circuits when `Warn` logging is disabled. The gather is
/// guarded by [`log_gather_format_vec!`].
#[macro_export]
macro_rules! gather_warn {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = $crate::log_gather_format_vec!(
            $comm_expr; log::Level::Warn; $($args)*
        ) {
            for fs in fsv {
                log::warn!("{}", fs);
            }
        }
    };
}

/// Assert that two values are equal, returning an `anyhow` error if not.
///
/// # Description
/// Wraps `anyhow::ensure!(left == right, ...)` and attaches an
/// [`EnsureError::LR`] payload that includes the source file, line
/// number and the formatted left/right values.
///
/// # Examples
/// \```
/// ensure_eq!(send_counts.len(), comm.size() as usize);
/// \```
#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr $(,)?) => {{
        let lv = ($left);
        let rv = ($right);
        anyhow::ensure!(
            lv == rv,
            $crate::log::EnsureError::LR(file!(), line!(), lv, rv)
        );
    }};
}

/// Assert that a boolean condition holds, returning an `anyhow`
/// error if not.
///
/// # Description
/// Wraps `anyhow::ensure!(cond, ...)` and attaches an
/// [`EnsureError::C`] payload pointing to the source file and line.
///
/// # Examples
/// \```
/// ensure!(part.local_size() == t_out.len());
/// \```
#[macro_export]
macro_rules! ensure {
    ($cond:expr $(,)?) => {{
        anyhow::ensure!(
            ($cond),
            $crate::log::EnsureError::C(file!(), line!(), 0, 0)
        );
    }};
}
