use askama::Template;
use clap::Parser;
use log::{error, info};
use std::{
    fs,
    io::BufWriter,
    path::{Path, PathBuf},
    process,
    sync::OnceLock,
};

use crate::config::{Config, ConfigBuilder};

mod config;

static CONFIG: OnceLock<Config> = OnceLock::new();

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

#[derive(serde::Deserialize, Debug)]
struct CudaConfig {
    kernels: Vec<Kernel>,
    setup: SetupFunction,
    #[serde(default)]
    free: FreeFunction,
}

#[derive(serde::Deserialize, Debug)]
struct Kernel {
    #[serde(default = "Kernel::default_return_type")]
    return_type: String,
    name: String,
    #[serde(default)]
    args: Vec<FunctionArg>,
    blocks: u32,
    threads: u32,
    stream: String,
}
impl Kernel {
    fn default_return_type() -> String {
        String::from("void")
    }
}

#[derive(serde::Deserialize, Debug)]
struct SetupFunction {
    #[serde(default = "SetupFunction::default_return_type")]
    return_type: String,
    #[serde(default = "SetupFunction::default_name")]
    name: String,
    #[serde(default)] // Default to no arguments
    args: Vec<FunctionArg>,
}
impl SetupFunction {
    fn default_return_type() -> String {
        String::from("void")
    }
    fn default_name() -> String {
        String::from("setup")
    }
}

#[derive(serde::Deserialize, Debug)]
struct FreeFunction {
    #[serde(default = "FreeFunction::default_return_type")]
    return_type: String,
    #[serde(default = "FreeFunction::default_name")]
    name: String,
}

impl Default for FreeFunction {
    fn default() -> Self {
        Self {
            return_type: Self::default_return_type(),
            name: Self::default_name(),
        }
    }
}
impl FreeFunction {
    fn default_return_type() -> String {
        String::from("void")
    }
    fn default_name() -> String {
        String::from("free_data")
    }
}

#[derive(serde::Deserialize, Debug)]
struct FunctionArg {
    name: String,
    #[serde(rename = "type")]
    datatype: String,
}

fn main() {
    // Check for loggin env var and default it if it is not present
    if let Err(_) = std::env::var("RUST_LOG") {
        // Single threaded space guarantees safety here
        unsafe { std::env::set_var("RUST_LOG", "info"); }
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
    let cuda_config: CudaConfig = {
        let path = global_config.get_config_file_path();
        let content = fs::read_to_string(path).expect("Failed to read config file");
        serde_yml::from_str(&content).expect("Failed to parse config file")
    };

    let header_generator = HeaderTemplate {
        config: &cuda_config,
    };
    let output_dir = global_config.get_output_dir();
    if !fs::exists(output_dir).unwrap_or(false) {
        let _ = fs::create_dir_all(output_dir);
    }
    let header_path = output_dir.to_path_buf().join("generated_kernels.h");
    let header_file = fs::File::create(&header_path).unwrap();
    let mut writer = BufWriter::new(header_file);
    let _ = header_generator.write_into(&mut writer);

    let canon_header_path = header_path.canonicalize().unwrap();
    let runner_generator = RunnerTemplate {
        config: &cuda_config,
        header_path: canon_header_path.to_str().unwrap(),
    };
    let runner_path = output_dir.to_path_buf().join("generated_runner.cu");
    let runner_file = fs::File::create(&runner_path).unwrap();
    let mut writer = BufWriter::new(runner_file);
    let _ = runner_generator.write_into(&mut writer);

    let binary_path = output_dir.to_path_buf().join("harness");
    let nvcc_output = process::Command::new("nvcc")
        .arg("-O3")
        .arg("-lineinfo")
        .arg(runner_path)
        .arg(args.input_file_path)
        .arg("-o")
        .arg(binary_path)
        .output();
    match nvcc_output {
        Ok(_) => info!("Compiled binary"),
        Err(e) => error!("Error in NVCC: {e}"),
    }
}
