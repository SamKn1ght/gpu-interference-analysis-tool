use askama::Template;
use clap::Parser;
use log::{error, info};
use polars::{prelude::*, series::Series};
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
    data::{
        CudaApiTrace, CudaGpuTrace, collect_all_array, lazy_load_api_trace_dataframe,
        lazy_load_gpu_trace_dataframe,
    },
    views::{GpuPipelineView, PairedKernelView},
};

mod config;
mod cuda;
mod data;
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
    let generated_dir = global_config.new_output_file("generated");
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

    for pair in PairedKernelView::iter_unique_kernel_pairs(&cuda_config) {
        // Generate files
        let runner_generator = PairedRunner {
            config: &cuda_config,
            header_path: canon_header_path.to_str().unwrap(),
            pair: &pair,
        };

        let pair_dir = global_config.new_output_file(pair.to_pair_name());
        let _ = fs::create_dir_all(&pair_dir);
        let pair_generated_dir = pair_dir.join("generated");
        let _ = fs::create_dir_all(&pair_generated_dir);
        let runner_path = pair_generated_dir.to_path_buf().join(RUNNER_FILE_SUFFIX);
        let binary_path = pair_generated_dir.to_path_buf().join("harness.bin");

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

        let nsys_output_file = pair_dir.join("report");
        let mut nsys_command = process::Command::new("nsys");
        nsys_command
            .arg("profile")
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

        let mut nsys_stats_command = process::Command::new("nsys");
        nsys_stats_command
            .arg("stats")
            .arg("--report")
            .arg("cuda_gpu_trace,cuda_api_trace")
            .arg("--format")
            .arg("csv")
            .arg("--output")
            .arg(format!("{}", nsys_output_file.to_string_lossy()))
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

        let api_trace_file = nsys_output_file.with_file_name(format!(
            "{}_cuda_api_trace.csv",
            nsys_output_file
                .file_name()
                .and_then(|s| s.to_str())
                .expect("Nsys output file should have some filename component")
        ));
        let gpu_trace_file = nsys_output_file.with_file_name(format!(
            "{}_cuda_gpu_trace.csv",
            nsys_output_file
                .file_name()
                .and_then(|s| s.to_str())
                .expect("Nsys output file should have some filename component")
        ));

        let calc_start = std::time::Instant::now();

        let api_data = lazy_load_api_trace_dataframe(&api_trace_file).unwrap();
        let gpu_data = lazy_load_gpu_trace_dataframe(&gpu_trace_file).unwrap();

        let kernel_names_series = Series::new("Kernel Names".into(), pair.get_kernel_names());
        let user_kernels_filtered = gpu_data
            .clone()
            .filter(col(CudaGpuTrace::NAME).is_in(lit(kernel_names_series).implode(), false));
        let final_executions_data = user_kernels_filtered
            .clone()
            .group_by([col(CudaGpuTrace::NAME)])
            .agg([col(CudaGpuTrace::END).max().alias("Max End")]);

        // Extract the last kernel execution timestamps
        let kernel_a_last_execution_row = final_executions_data
            .clone()
            .filter(col(CudaGpuTrace::NAME).eq(lit(pair.get_kernel_names()[0])));
        let kernel_b_last_execution_row = final_executions_data
            .clone()
            .filter(col(CudaGpuTrace::NAME).eq(lit(pair.get_kernel_names()[1])));
        let last_executions = LazyFrame::collect_all_with_engine(
            vec![
                kernel_a_last_execution_row.logical_plan,
                kernel_b_last_execution_row.logical_plan,
            ],
            Engine::Auto,
            OptFlags::default(),
        )
        .unwrap();
        let kernel_a_last_end = lit(last_executions[0]
            .column("Max End")
            .unwrap()
            .get(0)
            .unwrap()
            .try_extract::<i64>()
            .unwrap())
        .cast(DataType::Duration(TimeUnit::Nanoseconds));
        let kernel_b_last_end = lit(last_executions[1]
            .column("Max End")
            .unwrap()
            .get(0)
            .unwrap()
            .try_extract::<i64>()
            .unwrap())
        .cast(DataType::Duration(TimeUnit::Nanoseconds));

        let overlapping_kernel_condition = col(CudaGpuTrace::NAME)
            .cast(DataType::String)
            .eq(lit(pair.get_kernel_names()[0]))
            .and(col(CudaGpuTrace::START).lt(kernel_b_last_end))
            .or(col(CudaGpuTrace::NAME)
                .cast(DataType::String)
                .eq(lit(pair.get_kernel_names()[1]))
                .and(col(CudaGpuTrace::START).lt(kernel_a_last_end)));
        let concurrent_user_kernels = user_kernels_filtered
            .clone()
            .filter(overlapping_kernel_condition.clone());
        let kernel_count_summary = user_kernels_filtered
            .clone()
            .group_by([col(CudaGpuTrace::NAME)])
            .agg([
                len().alias("Total Count"),
                overlapping_kernel_condition
                    .clone()
                    .sum()
                    .alias("Overlapping Kernels"),
            ])
            .with_columns([
                (col("Total Count") - col("Overlapping Kernels")).alias("Excluded Kernels")
            ]);

        let queue_overhead_data = api_data
            .clone()
            .join(
                gpu_data.clone(),
                [col(CudaApiTrace::CORR_ID)],
                [col(CudaGpuTrace::CORR_ID)],
                JoinArgs::new(JoinType::Inner),
            )
            .with_column((col("Start (ns)_right") - col("Start (ns)")).alias(LAUNCH_LATENCY_STR))
            .select([
                col("Name").alias("Api Name"),
                col("Name_right").alias("Gpu Name"),
                col(CudaApiTrace::CORR_ID),
                col(LAUNCH_LATENCY_STR),
            ]);

        let [
            final_executions_data,
            queue_overhead_data,
            concurrent_user_kernels,
            kernel_count_summary,
        ] = collect_all_array([
            final_executions_data,
            queue_overhead_data,
            concurrent_user_kernels,
            kernel_count_summary,
        ])
        .unwrap();

        let calc_end = std::time::Instant::now();

        println!("Queue Launch Overhead: {}", queue_overhead_data.head(Some(10)));
        // println!("{}", final_executions_data);
        println!("Concurrent kernel [Head 5]: {}", concurrent_user_kernels.head(Some(5)));
        println!("Kernel Inc/Ex Count: {}", kernel_count_summary);
        println!("Polars calculations took {:#?}", calc_end - calc_start);
    }
}

const LAUNCH_LATENCY_STR: &str = "Launch Latency (ns)";
