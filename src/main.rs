use clap::Parser;
use std::{fs, path::PathBuf, sync::OnceLock};

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
}

#[derive(serde::Deserialize, Debug)]
struct CudaConfig {
    kernels: Vec<Kernel>,
    setup: SetupFunction,
}

#[derive(serde::Deserialize, Debug)]
struct Kernel {
    #[serde(default = "Kernel::default_return_type")]
    return_type: String,
    name: String,
    #[serde(default)]
    args: Vec<String>,
    blocks: u32,
    threads: u32,
}
impl Kernel {
    fn default_return_type() -> String {
        String::from("void")
    }
}

#[derive(serde::Deserialize, Debug)]
struct SetupFunction {
    name: String,
    #[serde(default)] // Default to no arguments
    args: Vec<String>,
}

fn main() {
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

    println!("{:?}", args);
    println!("{:?}", CONFIG.get().unwrap());
    println!("{:#?}", cuda_config);
}
