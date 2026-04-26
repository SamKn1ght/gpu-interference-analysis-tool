use askama::Template;
use clap::Parser;
use log::{error, info};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    process,
    sync::OnceLock,
};

use crate::{
    config::{Config, ConfigBuilder},
    cuda::{CudaConfig, DelayMethod},
    views::{GpuPipelineView, PairedKernelView},
};

mod config;
mod cuda;
mod views;

static CONFIG: OnceLock<Config> = OnceLock::new();

const KERNEL_HEADER_SUFFIX: &str = "generated_kernels.h";
const RUNNER_FILE_SUFFIX: &str = "generated_runner.cu";

#[derive(Parser, Debug)]
#[command(
    name = env!("CARGO_PKG_NAME"),
    version = env!("CARGO_PKG_VERSION")
)]
struct Args {
    /// Input cuda file
    #[arg()]
    input_file_path: String,
    /// Config file path
    #[arg(short, long = "config")]
    config_file_path: Option<PathBuf>,
    /// Output directory
    #[arg(short, long = "out")]
    output_dir: Option<PathBuf>,
}

#[derive(Template)]
#[template(path = "kernels.h.jinja")]
struct HeaderTemplate<'a> {
    config: &'a CudaConfig,
}

#[derive(Template)]
#[template(path = "runner.cu.jinja")]
struct RunnerTemplate<'a> {
    config: &'a CudaConfig,
    header_path: &'a str,
    pipelines: Vec<GpuPipelineView<'a>>,
}

#[derive(Template)]
#[template(path = "paired_runner.cu.jinja")]
struct PairedRunner<'a> {
    config: &'a CudaConfig,
    header_path: &'a str,
    pair: &'a PairedKernelView<'a>,
}

fn main() {
    // Check for loggin env var and default it if it is not present
    if let Err(_) = std::env::var("RUST_LOG") {
        // Single threaded space guarantees safety here
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    }
    env_logger::init();

    let args = Args::parse();

    let mut config_builder = ConfigBuilder::new();
    config_builder.input_file_path(&args.input_file_path);
    if let Some(path) = &args.config_file_path {
        config_builder.config_file_path(path);
    }
    let _ = CONFIG.set(
        config_builder
            .build()
            .expect("Fields should have been validated by this point"),
    );

    let global_config = CONFIG.get().expect("Config should be intitialised");
    let cuda_config = {
        let path = global_config.get_config_file_path();
        let content = fs::read_to_string(path).expect("Failed to read config file");
        let mut config: CudaConfig =
            serde_yml::from_str(&content).expect("Failed to parse config file");
        let _ = config.validate();
        config
    };

    let header_generator = HeaderTemplate {
        config: &cuda_config,
    };
    let generated_dir = global_config
        .get_output_dir()
        .to_path_buf()
        .join("generated");
    if !fs::exists(&generated_dir).unwrap_or(false) {
        let _ = fs::create_dir_all(&generated_dir);
    }

    // Generate header for the full user implementation file

    let header_path = generated_dir.to_path_buf().join(KERNEL_HEADER_SUFFIX);
    let header_file = fs::File::create(&header_path).unwrap();
    let mut writer = BufWriter::new(header_file);
    let _ = header_generator.write_into(&mut writer);
    let _ = writer.flush();
    let header_path = header_path
        .canonicalize()
        .expect("Header path should exist");

    let canon_header_path = header_path.canonicalize().unwrap();

    // Generate runner files for pairings

    let runner_path = generated_dir.to_path_buf().join(RUNNER_FILE_SUFFIX);
    let binary_path = generated_dir.to_path_buf().join("harness.bin");
    for pair in PairedKernelView::iter_unique_kernel_pairs(&cuda_config) {
        // Generate files
        let runner_generator = PairedRunner {
            config: &cuda_config,
            header_path: canon_header_path.to_str().unwrap(),
            pair: &pair,
        };

        let runner_file = fs::File::create(&runner_path).unwrap();
        let mut writer = BufWriter::new(runner_file);
        let _ = runner_generator.write_into(&mut writer);
        let _ = writer.flush();
        let runner_path = runner_path
            .canonicalize()
            .expect("Runner path should exist");

        // Build with NVCC

        let mut nvcc_command = process::Command::new("nvcc");
        nvcc_command
            .arg("-rdc=true")
            .arg("-I")
            .arg(
                &generated_dir
                    .canonicalize()
                    .expect("Output directory should exist"),
            )
            .arg("-O3")
            .arg("-lineinfo")
            .arg(
                PathBuf::from(&args.input_file_path)
                    .canonicalize()
                    .expect("User provided file path should exist"),
            )
            .arg(runner_path)
            .arg("-o")
            .arg(&binary_path);

        match nvcc_command.output() {
            Ok(out) => {
                if out.status.success() {
                    info!("Compiled binary {}", binary_path.to_string_lossy())
                } else {
                    error!("Error in NVCC: {}", String::from_utf8_lossy(&out.stderr));
                }
            }
            Err(e) => error!("Error in NVCC: {e}"),
        }

        // Run nsys command on generated binary

        let nsys_output_file = global_config
            .get_output_dir()
            .to_path_buf()
            .join(format!("{}", pair.to_pair_name()));
        let mut nsys_command = process::Command::new("nsys");
        nsys_command
            .arg("profile")
            // .arg("--stats=true")
            .arg("-o")
            .arg(format!("{}", nsys_output_file.to_string_lossy()))
            .arg(format!("{}", binary_path.to_string_lossy()));

        match nsys_command.output() {
            Ok(out) => {
                if out.status.success() {
                    info!("Nsys completed for {}", pair.to_pair_name());
                } else {
                    error!("Error from NSYS: {}", String::from_utf8_lossy(&out.stderr));
                }
            }
            Err(e) => error!("Error running NSYS: {e}"),
        }

        let csv_output_file = global_config
            .get_output_dir()
            .to_path_buf()
            .join(format!("{}.csv", pair.to_pair_name()));
        let mut nsys_stats_command = process::Command::new("nsys");
        nsys_stats_command
            .arg("stats")
            .arg("--report")
            .arg("cuda_gpu_trace,cuda_api_trace")
            .arg("--format")
            .arg("csv")
            .arg("--output")
            .arg(format!("{}", csv_output_file.to_string_lossy()))
            .arg(format!("{}.nsys-rep", nsys_output_file.to_string_lossy()));

        match nsys_stats_command.output() {
            Ok(out) => {
                if out.status.success() {
                    info!("Nsys stats completed for {}", pair.to_pair_name());
                } else {
                    error!(
                        "Error from NSYS stats: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                }
            }
            Err(e) => error!("Error running NSYS stats: {e}"),
        }
    }
}
