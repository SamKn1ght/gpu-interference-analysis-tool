use std::path::PathBuf;

use clap::Parser;

use crate::config::ConfigBuilder;

mod config;

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
    #[arg(short, long="config")]
    config_file_path: Option<PathBuf>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let mut config_builder = ConfigBuilder::new();
    config_builder.input_file_path(&args.input_file_path);
    if let Some(path) = &args.config_file_path {
        config_builder.config_file_path(path);
    }
    let config = config_builder.build();

    println!("{:?}", args);
    println!("{:?}", config);
}
