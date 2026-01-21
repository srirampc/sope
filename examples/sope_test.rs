use anyhow::Result;
use clap::Parser;
use mpi::traits::{Communicator, Equivalence, Root};
use serde::{Deserialize, Serialize};

use sope::{comm::WorldComm, cond_error, cond_info, shift::right_shift};

/// Parensnet:: Parallel Ensembl Gene Network Construction
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct CLIArgs {
    /// Path to input YAML file with all the arguments
    config: std::path::PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Equivalence)]
pub struct RT {
    vx: i32,
    fx: f32,
}

impl Default for RT {
    fn default() -> Self {
        RT { vx: 0, fx: 0.0 }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InArgs {
    // Mandatory Fileds
    //  - Files/Paths
    data: Vec<RT>,
}

fn parse_args(mcx: &WorldComm, args: &CLIArgs) -> Result<InArgs> {
    match serde_saphyr::from_str::<InArgs>(&std::fs::read_to_string(&args.config)?) {
        Ok(wargs) => {
            cond_info!(mcx.is_root(); "Parsed successfully: {:?}", wargs);
            Ok(wargs)
        }
        Err(err) => {
            cond_error!(mcx.is_root(); "Failed to parse YAML: {}", err);
            Err(anyhow::anyhow!(err))
        }
    }
}

fn run(mcx: &WorldComm, args: CLIArgs) -> Result<()> {
    env_logger::try_init()?;
    let wargs = parse_args(mcx, &args)?;
    let pvx = &wargs.data[mcx.rank as usize];
    let root_process = mcx.comm.process_at_rank(0);
    let pvxv = vec![pvx.clone()];
    if mcx.is_root() {
        let mut pvx_vec = vec![RT::default(); mcx.size as usize];
        root_process.gather_into_root(&pvxv, &mut pvx_vec);
        log::info!("ROOT RTX {:?}", pvx_vec);
    } else {
        root_process.gather_into(&pvxv);
    }
    let rtx = right_shift(pvx, &mcx.comm);
    if mcx.is_root() {
        let mut rtx_vec = vec![RT::default(); mcx.size as usize];
        root_process.gather_into_root(&rtx, &mut rtx_vec);
        log::info!("ROOT RTX {:?}", rtx_vec);
    } else {
        root_process.gather_into(&rtx);
    }

    Ok(())
}

fn main() {
    let comm_ifx = WorldComm::init();
    match CLIArgs::try_parse() {
        Ok(args) => {
            let _ = run(&comm_ifx, args);
        }
        Err(err) => {
            if comm_ifx.rank == 0 {
                let _ = err.print();
            };
        }
    };
}
