use std::fmt::{Debug, Display};
use thiserror::Error;
#[derive(Error, Debug)]
pub enum EnsureError<F: Debug + Display, T: Debug + Display, L: Debug, R: Debug> {
    #[error("ensure failed at {0}:{1}")]
    C(F, T, L, R),
    #[error("ensure `left == right` failed at {0}:{1} :: left:({2:?}), right:({3:?})")]
    LR(F, T, L, R),
}

#[macro_export]
macro_rules! cond_println {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            println!($($args)*)
        }
    };
}

#[macro_export]
macro_rules! cond_eprintln {
    ($cond_expr: expr; $($args:tt)* ) => {
        if $cond_expr {
            eprintln!($($args)*)
        }
    };
}

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

#[macro_export]
macro_rules! gather_format_vec {
    ($comm_expr: expr; $($args:tt)* ) => {{
        let frs = format!($($args)*);
        let s = if frs.len() > 0 {
            use ::mpi::topology::Communicator;
            format!(
                "[{}::{}] {}", ($comm_expr).rank(),
                ::chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                frs
            )
        } else {
            frs
        };
        $crate::collective::gather_strings(s, 0, ($comm_expr))
    }};
}

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

#[macro_export]
macro_rules! gather_format {
    ($comm_expr: expr; $($args:tt)* ) => {
        match $crate::gather_format_vec!($comm_expr; $($args)*) {
            ::anyhow::Result::Ok(rsv) => {
                rsv.map(|sv| sv.join("\n"))
            }
            ::anyhow::Result::Err(err) => {
                Some(err.to_string())
            }
        }
    };
}

#[macro_export]
macro_rules! gather_println {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fs) = $crate::gather_format!($comm_expr; $($args)*) {
            eprintln!("{:?}", fs);
        }
    };
}

#[macro_export]
macro_rules! gather_eprintln {
    ($comm_expr: expr;$($args:tt)* ) => {
        if let fs = $crate::gather_format!($comm_expr; $($args)*) {
            eprintln!("{:?}", fs);
        }
    };
}

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

#[macro_export]
macro_rules! gather_error {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = sope::log_gather_format_vec!(
            $comm_expr; log::Level::Error; $($args)*
        ) {
            for fs in fsv {
                log::error!("{}", fs);
            }
        }
    }
}

#[macro_export]
macro_rules! gather_debug {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = sope::log_gather_format_vec!(
            $comm_expr; log::Level::Debug; $($args)*
        ) {
            for fs in fsv {
                log::debug!("{}", fs);
            }
        }
    }
}

#[macro_export]
macro_rules! gather_warn {
    ($comm_expr: expr; $($args:tt)* ) => {
        if let Some(fsv) = sope::log_gather_format_vec!(
            $comm_expr; log::Level::Warn; $($args)*
        ) {
            for fs in fsv {
                log::warn!("{}", fs);
            }
        }
    };
}

#[macro_export]
macro_rules! ensure_eq {
    ($left:expr, $right:expr $(,)?) => {{
        let lv = ($left);
        let rv = ($right);
        anyhow::ensure!(
            lv == rv,
            sope::log::EnsureError::LR(file!(), line!(), lv, rv)
        );
    }};
}

#[macro_export]
macro_rules! ensure {
    ($cond:expr $(,)?) => {{
        anyhow::ensure!(($cond), sope::log::EnsureError::C(file!(), line!(), 0, 0));
    }};
}
