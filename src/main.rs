use askama::Template;
use clap::Parser;
use log::{error, info};
use std::{fs, io::{BufWriter, Write}, path::PathBuf, process, sync::OnceLock};

use crate::{
    config::{Config, ConfigBuilder},
    cuda::CudaConfig,
};

mod config;
mod cuda;

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
    let output_dir = global_config.get_output_dir();
    if !fs::exists(output_dir).unwrap_or(false) {
        let _ = fs::create_dir_all(output_dir);
    }
    let header_path = output_dir.to_path_buf().join(KERNEL_HEADER_SUFFIX);
    let header_file = fs::File::create(&header_path).unwrap();
    let mut writer = BufWriter::new(header_file);
    let _ = header_generator.write_into(&mut writer);
    let _ = writer.flush();
    let header_path = header_path
        .canonicalize()
        .expect("Header path should exist");

    let canon_header_path = header_path.canonicalize().unwrap();
    let runner_generator = RunnerTemplate {
        config: &cuda_config,
        header_path: canon_header_path.to_str().unwrap(),
    };
    let runner_path = output_dir.to_path_buf().join(RUNNER_FILE_SUFFIX);
    let runner_file = fs::File::create(&runner_path).unwrap();
    let mut writer = BufWriter::new(runner_file);
    let _ = runner_generator.write_into(&mut writer);
    let _ = writer.flush();
    let runner_path = runner_path
        .canonicalize()
        .expect("Runner path should exist");

    let binary_path = output_dir.to_path_buf().join("harness.bin");
    let mut nvcc_command = process::Command::new("nvcc");
    nvcc_command
        .arg("-rdc=true")
        .arg("-I")
        .arg(&output_dir.canonicalize().expect("Output directory should exist"))
        .arg("-O3")
        .arg("-lineinfo")
        .arg(
            PathBuf::from(args.input_file_path)
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
}
