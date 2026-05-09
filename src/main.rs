use askama::Template;
use clap::Parser;
use log::{debug, error, info};
use polars::{prelude::*, series::Series};
use std::{
    fs::{self},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process,
    sync::OnceLock,
};

use crate::{
    config::{Config, ConfigBuilder},
    cuda::{CudaConfig, DelayMethod, Kernel, Stream},
    data::{
        CudaApiTrace, CudaGpuTrace, NcuData, collect_all_array, get_gpu_duration_summary,
        get_pivoted_table_for_attribute, get_system_latency_summary, lazy_load_api_trace_dataframe,
        lazy_load_gpu_trace_dataframe, lazy_load_ncu_dataframe, pivot_ncu_data, write_to_csv,
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

trait RunnerTemplate {}

#[derive(Template)]
#[template(path = "single_runner.cu.jinja")]
struct SingleRunner<'a> {
    config: &'a CudaConfig,
    header_path: &'a str,
    kernel: &'a Kernel,
    stream: &'a Stream,
}
impl<'a> RunnerTemplate for SingleRunner<'a> {}

#[derive(Template)]
#[template(path = "paired_runner.cu.jinja")]
struct PairedRunner<'a> {
    config: &'a CudaConfig,
    header_path: &'a str,
    pair: &'a PairedKernelView<'a>,
}
impl<'a> RunnerTemplate for PairedRunner<'a> {}

#[inline(never)]
fn compile_runner<G>(
    generator: G,
    root_path: &Path,
    header_dir: &Path,
    user_file_path: &Path,
) -> PathBuf
where
    G: Template + RunnerTemplate,
{
    let _ = fs::create_dir_all(&root_path);
    let generated_dir = root_path.join("generated");
    let _ = fs::create_dir_all(&generated_dir);
    let runner_path = generated_dir.join(RUNNER_FILE_SUFFIX);
    let binary_path = generated_dir.join("harness.bin");

    let runner_file = fs::File::create(&runner_path).unwrap();
    let mut writer = BufWriter::new(runner_file);
    let _ = Template::write_into(&generator, &mut writer);
    let _ = writer.flush();
    let runner_path = runner_path
        .canonicalize()
        .expect("Runner path should exist");

    let mut nvcc_command = process::Command::new("nvcc");
    nvcc_command
        .arg("-rdc=true")
        .arg("-I")
        .arg(
            &header_dir
                .canonicalize()
                .expect("Output directory should exist"),
        )
        .arg("-O3")
        .arg("-lineinfo")
        .arg(
            user_file_path
                .to_path_buf()
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
                error!(
                    "Error in NVCC stdout: {}, stderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        Err(e) => error!("Error in NVCC: {e}"),
    }

    binary_path
}
#[inline(never)]
fn run_nsys(binary_path: &Path, output_file: &Path, trial_name: &str) {
    let mut nsys_command = process::Command::new("nsys");
    nsys_command
        .arg("profile")
        .arg("--trace")
        .arg("cuda")
        .arg("-o")
        .arg(format!("{}", output_file.to_string_lossy()))
        .arg(format!("{}", binary_path.to_string_lossy()));

    match nsys_command.output() {
        Ok(out) => {
            if out.status.success() {
                info!("Nsys completed for {}", trial_name);
            } else {
                error!(
                    "Error in NSYS stdout: {}, stderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        Err(e) => error!("Error running NSYS: {e}"),
    }
}
fn get_nsys_trace_paths(nsys_report_file: &Path, trial_name: &str) -> (PathBuf, PathBuf) {
    let mut nsys_stats_command = process::Command::new("nsys");
    nsys_stats_command
        .arg("stats")
        .arg("--report")
        .arg("cuda_gpu_trace,cuda_api_trace")
        .arg("--format")
        .arg("csv")
        .arg("--output")
        .arg(format!("{}", nsys_report_file.to_string_lossy()))
        .arg(format!("{}.nsys-rep", nsys_report_file.to_string_lossy()));

    match nsys_stats_command.output() {
        Ok(out) => {
            if out.status.success() {
                info!("Nsys stats completed for {}", trial_name);
            } else {
                error!(
                    "Error in NSYS stats stdout: {}, stderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        Err(e) => error!("Error running NSYS stats: {e}"),
    }

    let api_trace_file = nsys_report_file.with_file_name(format!(
        "{}_cuda_api_trace.csv",
        nsys_report_file
            .file_name()
            .and_then(|s| s.to_str())
            .expect("Nsys output file should have some filename component")
    ));
    let gpu_trace_file = nsys_report_file.with_file_name(format!(
        "{}_cuda_gpu_trace.csv",
        nsys_report_file
            .file_name()
            .and_then(|s| s.to_str())
            .expect("Nsys output file should have some filename component")
    ));
    (api_trace_file, gpu_trace_file)
}
/// Runs the ncu command for a singular kernel execution `trial_name` returning the CSV file from
/// the stdout as a cursor if the command is successful
#[inline(never)]
fn run_ncu(binary_path: &Path, output_file: &Path, trial_name: &str) {
    let mut ncu_command = process::Command::new("ncu");
    ncu_command
        .args([
            "--section",
            "SpeedOfLight",
            "--section",
            "Occupancy",
            "--section",
            "MemoryWorkloadAnalysis",
            "--apply-rules",
            "no",
            "-c",
            "1",
            "-s",
            "1",
            "-k",
            trial_name,
            "--csv",
            "--page=details",
        ])
        .arg("-o")
        .arg(output_file)
        .arg(binary_path);

    match ncu_command.output() {
        Ok(out) => {
            if out.status.success() {
                info!("Ncu completed for {}", trial_name);
                let output = out.stdout;
                let mut bytes_to_skip = 0;

                for line in output.split(|x| *x == b'\n') {
                    if line.first() == Some(&b'=') {
                        bytes_to_skip += line.len() + 1;
                    } else {
                        break;
                    }
                }

                let out_slice = &output[bytes_to_skip..];
                let file = fs::File::create(output_file).unwrap();
                let mut writer = BufWriter::new(file);
                let _ = writer.write(out_slice);
                let _ = writer.flush();
            } else {
                error!(
                    "Error in NCU stdout: {}, stderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }
        Err(e) => {
            error!("Error running NCU: {e}");
        }
    }
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
    let user_input_file = PathBuf::from(&args.input_file_path);

    // Generate baseline data

    let baseline_path = global_config.new_output_file("baseline");
    let mut gpu_kernel_rows = Vec::with_capacity(cuda_config.kernels.len());
    let mut api_data_rows = Vec::with_capacity(cuda_config.kernels.len());
    let mut ncu_data_rows = Vec::with_capacity(cuda_config.kernels.len());
    for kernel in &cuda_config.kernels {
        let runner_generator = SingleRunner {
            config: &cuda_config,
            header_path: header_path.to_str().unwrap(),
            kernel,
            stream: kernel.get_stream(&cuda_config).unwrap(),
        };
        let single_dir = baseline_path.join(&kernel.name);
        let binary_path = compile_runner(
            runner_generator,
            &single_dir,
            &generated_dir,
            &user_input_file,
        );

        let nsys_output_file = single_dir.join("report");
        let ncu_output_file = single_dir.join("profile.csv");
        run_nsys(&binary_path, &nsys_output_file, &kernel.name);
        run_ncu(&binary_path, &ncu_output_file, &kernel.name);

        let ncu_dataframe = lazy_load_ncu_dataframe(&ncu_output_file).unwrap();
        println!("NCU Data: {}", ncu_dataframe.clone().collect().unwrap());
        let pivoted_ncu_dataframe = pivot_ncu_data(ncu_dataframe.clone());
        ncu_data_rows.push(pivoted_ncu_dataframe.clone());

        let (api_trace_file, gpu_trace_file) =
            get_nsys_trace_paths(&nsys_output_file, &kernel.name);

        let gpu_data = lazy_load_gpu_trace_dataframe(&gpu_trace_file).unwrap();
        let single_execution_duration_stats = gpu_data
            .clone()
            .filter(col(CudaGpuTrace::NAME).eq(lit(kernel.name.clone())))
            .sort([CudaGpuTrace::CORR_ID], SortMultipleOptions::default())
            .with_row_index("Instance ID", None)
            .filter(col("Instance ID").gt_eq(1));
        gpu_kernel_rows.push(get_gpu_duration_summary(single_execution_duration_stats));

        let api_data = lazy_load_api_trace_dataframe(&api_trace_file)
            .unwrap()
            .filter(
                (col(CudaApiTrace::NAME).eq(lit("cuKernelGetName")))
                    .or(col(CudaApiTrace::NAME).eq(lit("cudaLaunchKernel"))),
            )
            .join(
                gpu_data.clone().select([
                    col(CudaGpuTrace::CORR_ID),
                    col(CudaGpuTrace::NAME),
                    col(CudaGpuTrace::END),
                ]),
                [col(CudaApiTrace::CORR_ID)],
                [col(CudaGpuTrace::CORR_ID)],
                JoinArgs::new(JoinType::Left),
            )
            .rename(
                [
                    format!("{}_right", CudaGpuTrace::NAME),
                    format!("{}_right", CudaGpuTrace::END),
                ],
                ["Kernel Name", "Kernel End"],
                true,
            )
            .sort([CudaApiTrace::CORR_ID], SortMultipleOptions::default())
            .with_row_index("Instance ID", None)
            .with_column(col("Instance ID") / lit(2))
            .filter(col("Instance ID").gt_eq(1))
            .group_by([col("Instance ID")])
            .agg([
                col("Kernel Name").first_non_null(),
                col(CudaApiTrace::START).min().alias("Frame Start"),
                col("Kernel End").max().alias("Frame End"),
            ])
            .with_column((col("Frame End") - col("Frame Start")).alias("System Latency"))
            .with_column(
                lit(cuda_config.get_frame_budget().duration.as_nanos()).alias("Frame Allowance"),
            )
            .with_columns([
                col("System Latency")
                    .gt(col("Frame Allowance"))
                    .alias("Missed Deadline"),
                (col("System Latency").cast(DataType::Float64)
                    / col("Frame Allowance").cast(DataType::Float64))
                .alias("Frame Allowance Usage"),
            ]);
        api_data_rows.push(get_system_latency_summary(api_data));
    }
    let duration_summaries = concat(&gpu_kernel_rows, UnionArgs::default()).unwrap();
    let system_latency_summaries = concat(&api_data_rows, UnionArgs::default()).unwrap();
    let ncu_profile_data = concat(&ncu_data_rows, UnionArgs::default()).unwrap();
    println!(
        "Duration Summary: {}",
        duration_summaries.clone().collect().unwrap()
    );
    println!(
        "System Latency Summary: {}",
        system_latency_summaries.clone().collect().unwrap()
    );
    println!(
        "Ncu Profiling Data: {}",
        ncu_profile_data.clone().collect().unwrap()
    );

    // Generate runner files for pairings

    let mut pair_duration_change_rows =
        Vec::with_capacity(&cuda_config.kernels.len() * (&cuda_config.kernels.len() - 1) / 2);
    let mut pair_system_latency_rows =
        Vec::with_capacity(&cuda_config.kernels.len() * (&cuda_config.kernels.len() - 1) / 2);
    let mut pair_ncu_profile_rows =
        Vec::with_capacity(&cuda_config.kernels.len() * (&cuda_config.kernels.len() - 1) / 2);
    for pair in PairedKernelView::iter_unique_kernel_pairs(&cuda_config) {
        // Generate files
        let runner_generator = PairedRunner {
            config: &cuda_config,
            header_path: canon_header_path.to_str().unwrap(),
            pair: &pair,
        };

        let pair_dir = global_config.new_output_file(pair.to_pair_name());
        let binary_path = compile_runner(
            runner_generator,
            &pair_dir,
            &generated_dir,
            &user_input_file,
        );

        // Run nsys command on generated binary

        let nsys_output_file = pair_dir.join("report");
        run_nsys(&binary_path, &nsys_output_file, &pair.to_pair_name());
        let (api_trace_file, gpu_trace_file) =
            get_nsys_trace_paths(&nsys_output_file, &pair.to_pair_name());

        let calc_start = std::time::Instant::now();

        let gpu_data = lazy_load_gpu_trace_dataframe(&gpu_trace_file).unwrap();
        let api_data = lazy_load_api_trace_dataframe(&api_trace_file).unwrap();

        let sum_contested_resource_names = [
            "Sum Bandwidth",
            "Sum Mem Busy",
            "Sum Compute Throughput",
            "Sum L2 Throughput",
            "Sum L1 Throughput",
        ];
        let sum_contested_resource_display_names = [
            "Memory Bandwidth",
            "Memory Busy",
            "Compute Units",
            "L2 Cache",
            "L1 Cache",
        ];
        let ncu_kernels_profiles = df!(
            "Kernel Name" => [pair.get_kernel_names()[0], pair.get_kernel_names()[1]],
            "Opposing Kernel" => [pair.get_kernel_names()[1], pair.get_kernel_names()[0]])
        .unwrap()
        .lazy()
        .join(
            ncu_profile_data.clone(),
            [col("Kernel Name")],
            [col(NcuData::KERNEL_NAME)],
            JoinArgs::new(JoinType::Inner),
        )
        .join(
            ncu_profile_data.clone(),
            [col("Opposing Kernel")],
            [col(NcuData::KERNEL_NAME)],
            JoinArgs::new(JoinType::Inner),
        )
        .with_columns([
            (col("Achieved Occupancy") + col("Achieved Occupancy_right")).alias("Sum Occupancy"),
            (col("Max Bandwidth") + col("Max Bandwidth_right")).alias("Sum Bandwidth"),
            (col("Mem Busy") + col("Mem Busy_right")).alias("Sum Mem Busy"),
            (col("Compute (SM) Throughput") + col("Compute (SM) Throughput_right"))
                .alias("Sum Compute Throughput"),
            (col("L2 Cache Throughput") + col("L2 Cache Throughput_right"))
                .alias("Sum L2 Throughput"),
            (col("L1/TEX Cache Throughput") + col("L1/TEX Cache Throughput_right"))
                .alias("Sum L1 Throughput"),
        ])
        .with_columns([
            concat_list([
                when(col("Sum Occupancy").gt(90.0))
                    .then(lit("Occupancy"))
                    .otherwise(lit(NULL)),
                when(col("Sum Bandwidth").gt(90.0))
                    .then(lit("Memory Bandwidth"))
                    .otherwise(lit(NULL)),
                when(col("Sum Mem Busy").gt(90.0))
                    .then(lit("Memory Busy"))
                    .otherwise(lit(NULL)),
                when(col("Sum Compute Throughput").gt(90.0))
                    .then(lit("Compute Units"))
                    .otherwise(lit(NULL)),
                when(col("Sum L2 Throughput").gt(90.0))
                    .then(lit("L2 Cache"))
                    .otherwise(lit(NULL)),
                when(col("Sum L1 Throughput").gt(90.0))
                    .then(lit("L1 Cache"))
                    .otherwise(lit(NULL)),
            ])
            .unwrap()
            .list()
            .eval(col("").drop_nulls())
            .alias("Contested Resources"),
            concat_list(
                sum_contested_resource_names
                    .iter()
                    .map(|s| col(*s))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .list()
            .arg_max()
            .alias("Max Contention Index"),
        ])
        .with_column(
            sum_contested_resource_names
                .iter()
                .enumerate()
                .fold(
                    when(lit(false))
                        .then(lit(NULL))
                        .when(lit(false))
                        .then(lit(NULL)),
                    |acc, (i, _label)| {
                        acc.when(col("Max Contention Index").eq(lit(i as u32)))
                            .then(lit(sum_contested_resource_display_names[i]))
                    },
                )
                .otherwise(lit(NULL))
                .alias("Main Bottleneck"),
        );
        pair_ncu_profile_rows.push(ncu_kernels_profiles.clone());

        println!(
            "Joined pair ncu: {}",
            ncu_kernels_profiles.clone().collect().unwrap()
        );

        let kernel_names_series = Series::new("Kernel Names".into(), pair.get_kernel_names());
        let user_kernels_filtered = gpu_data
            .clone()
            .filter(col(CudaGpuTrace::NAME).is_in(lit(kernel_names_series).implode(), false))
            .sort([CudaGpuTrace::CORR_ID], SortMultipleOptions::default())
            .with_column(
                col(CudaGpuTrace::NAME)
                    .cum_count(false)
                    .over([CudaGpuTrace::NAME])
                    .alias("Instance ID"),
            )
            .filter(col("Instance ID").gt(lit(1)))
            .with_column(col("Instance ID") - lit(1));
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
        let last_executions =
            collect_all_array([kernel_a_last_execution_row, kernel_b_last_execution_row]).unwrap();
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
        let paired_duration_summary = get_gpu_duration_summary(concurrent_user_kernels.clone());

        let duration_slowdown_summary = paired_duration_summary
            .clone()
            .join(
                duration_summaries.clone(),
                [col(CudaGpuTrace::NAME)],
                [col(CudaGpuTrace::NAME)],
                JoinArgs::new(JoinType::Inner),
            )
            .with_columns([
                lit(pair.get_kernel_names()[0]).alias("Kernel A"),
                lit(pair.get_kernel_names()[1]).alias("Kernel B"),
                (col("Mean").cast(DataType::Float64) / col("Mean_right").cast(DataType::Float64))
                    .alias("Mean"),
                (col("95%").cast(DataType::Float64) / col("95%_right").cast(DataType::Float64))
                    .alias("95%"),
                (col("99%").cast(DataType::Float64) / col("99%_right").cast(DataType::Float64))
                    .alias("99%"),
                (col("Max").cast(DataType::Float64) / col("Max_right").cast(DataType::Float64))
                    .alias("Max"),
                (col("Coefficient of Variation").cast(DataType::Float64)
                    / col("Coefficient of Variation_right").cast(DataType::Float64))
                .alias("Coefficient of Variation"),
            ])
            .select([
                col("Kernel A"),
                col("Kernel B"),
                col(CudaGpuTrace::NAME),
                col("Mean"),
                col("95%"),
                col("99%"),
                col("Max"),
                col("Coefficient of Variation"),
            ]);
        pair_duration_change_rows.push(duration_slowdown_summary.clone());

        let system_latency_data = api_data
            .clone()
            .filter(
                (col(CudaApiTrace::NAME).eq(lit("cuKernelGetName")))
                    .or(col(CudaApiTrace::NAME).eq(lit("cudaLaunchKernel"))),
            )
            .join(
                gpu_data.clone(),
                [col(CudaApiTrace::CORR_ID)],
                [col(CudaGpuTrace::CORR_ID)],
                JoinArgs::new(JoinType::Left),
            )
            .rename(
                [
                    format!("{}_right", CudaGpuTrace::NAME),
                    format!("{}_right", CudaGpuTrace::END),
                ],
                ["Kernel Name", "Kernel End"],
                true,
            )
            .sort([CudaApiTrace::CORR_ID], SortMultipleOptions::default())
            .with_columns_seq([
                col("Kernel Name").fill_null_with_strategy(FillNullStrategy::Backward(None)),
                col("Kernel Name")
                    .cum_count(false)
                    .over(["Kernel Name"])
                    .alias("Instance ID"),
            ])
            .filter(col("Instance ID").gt(lit(1)))
            .with_column(col("Instance ID") - lit(1))
            .group_by_stable([col("Instance ID"), col("Kernel Name")])
            .agg([
                col(CudaApiTrace::START).min().alias("Kernel Frame Start"),
                col("Kernel End").max().alias("Kernel Frame End"),
            ])
            .with_columns([
                col("Kernel Frame End")
                    .max()
                    .over(["Instance ID"])
                    .alias("Frame End"),
                col("Kernel Frame Start")
                    .min()
                    .over(["Instance ID"])
                    .alias("Frame Start"),
            ])
            .with_columns([
                (col("Frame End") - col("Frame Start")).alias("System Latency"),
                (col("Kernel Frame End") - col("Kernel Frame Start")).alias("Kernel Latency"),
                lit(cuda_config.get_frame_budget().duration.as_nanos()).alias("Frame Allowance"),
            ])
            .with_columns([
                col("System Latency")
                    .gt(col("Frame Allowance"))
                    .alias("Missed Deadline"),
                (col("System Latency").cast(DataType::Float64)
                    / col("Frame Allowance").cast(DataType::Float64))
                .alias("System Frame Usage"),
                col("Kernel Latency")
                    .gt(col("Frame Allowance"))
                    .alias("Kernel Missed Deadline"),
                (col("Kernel Latency").cast(DataType::Float64)
                    / col("Frame Allowance").cast(DataType::Float64))
                .alias("Kernel Frame Usage"),
            ]);
        let system_latency_summary = get_system_latency_summary(system_latency_data.clone())
            .with_columns([
                lit(pair.get_kernel_names()[0]).alias("Kernel A"),
                lit(pair.get_kernel_names()[1]).alias("Kernel B"),
            ])
            .with_column(
                when(col(CudaApiTrace::NAME).eq(col("Kernel A")))
                    .then(col("Kernel B"))
                    .otherwise(col("Kernel A"))
                    .alias("Opposing Kernel"),
            );
        pair_system_latency_rows.push(system_latency_summary.clone());

        let [
            final_executions_data,
            concurrent_user_kernels,
            kernel_count_summary,
            paired_duration_summary,
            duration_slowdown_summary,
            system_latency_data,
            system_latency_summary,
        ] = collect_all_array([
            final_executions_data,
            concurrent_user_kernels,
            kernel_count_summary,
            paired_duration_summary,
            duration_slowdown_summary,
            system_latency_data,
            system_latency_summary,
        ])
        .unwrap();

        let calc_end = std::time::Instant::now();

        println!("API DATA: {}", system_latency_summary);

        // println!("{}", final_executions_data);
        // println!(
        //     "Concurrent kernel [Head 5]: {}",
        //     concurrent_user_kernels.head(Some(5))
        // );
        println!("Kernel Inc/Ex Count: {}", kernel_count_summary);
        println!("Duration Summary: {}", paired_duration_summary);
        println!("Duration Slowdown Summary: {}", duration_slowdown_summary);
        debug!("Polars calculations took {:#?}", calc_end - calc_start);
    }

    let paired_system_latency_summary = concat(pair_system_latency_rows, UnionArgs::default())
        .unwrap()
        .join(
            system_latency_summaries
                .clone()
                .select([
                    col(CudaApiTrace::NAME),
                    col("System Latency Mean"),
                    col("System Latency 99%"),
                ])
                .rename(
                    [
                        CudaApiTrace::NAME,
                        "System Latency Mean",
                        "System Latency 99%",
                    ],
                    ["Kernel A", "Kernel A Mean Latency", "Kernel A 99% Latency"],
                    true,
                ),
            [col("Kernel A")],
            [col("Kernel A")],
            JoinArgs::new(JoinType::Left),
        )
        .join(
            system_latency_summaries
                .clone()
                .select([
                    col(CudaApiTrace::NAME),
                    col("System Latency Mean"),
                    col("System Latency 99%"),
                ])
                .rename(
                    [
                        CudaApiTrace::NAME,
                        "System Latency Mean",
                        "System Latency 99%",
                    ],
                    ["Kernel B", "Kernel B Mean Latency", "Kernel B 99% Latency"],
                    true,
                ),
            [col("Kernel B")],
            [col("Kernel B")],
            JoinArgs::new(JoinType::Left),
        )
        .with_columns([
            (col("Kernel A Mean Latency") + col("Kernel B Mean Latency"))
                .alias("Mean Sequential Latency"),
            (col("Kernel A 99% Latency") + col("Kernel B 99% Latency"))
                .alias("99% Sequential Latency"),
        ])
        .group_by([CudaApiTrace::NAME, "Opposing Kernel"])
        .agg([
            col("Mean Sequential Latency").first_non_null(),
            col("99% Sequential Latency").first_non_null(),
            col("System Latency Mean").first_non_null(),
            col("System Latency 99%").first_non_null(),
        ])
        .with_columns([
            (col("Mean Sequential Latency").cast(DataType::Float64)
                / col("System Latency Mean").cast(DataType::Float64))
            .alias("Mean Concurrency Efficiency"),
            (col("99% Sequential Latency").cast(DataType::Float64)
                / col("System Latency 99%").cast(DataType::Float64))
            .alias("99% Concurrency Efficiency"),
        ]);
    let pivoted_mean_system_latency_summary = get_pivoted_table_for_attribute(
        paired_system_latency_summary.clone(),
        "Mean Concurrency Efficiency",
        "Kernel",
    );
    let pivoted_p99_system_latency_summary = get_pivoted_table_for_attribute(
        paired_system_latency_summary.clone(),
        "99% Concurrency Efficiency",
        "Kernel",
    );

    let paired_kernel_duration_summary = concat(pair_duration_change_rows, UnionArgs::default())
        .unwrap()
        .with_columns([
            col("99%").min().alias("99% Min"),
            col("99%").max().alias("99% Max"),
            col("Coefficient of Variation")
                .min()
                .alias("Coefficient of Variation Min"),
            col("Coefficient of Variation")
                .max()
                .alias("Coefficient of Variation Max"),
        ])
        .with_columns([
            ((col("99%") - col("99% Min")) / (col("99% Max") - col("99% Min"))).alias("Norm 99%"),
            ((col("Coefficient of Variation") - col("Coefficient of Variation Min"))
                / (col("Coefficient of Variation Max") - col("Coefficient of Variation Min")))
            .alias("Norm Coefficient of Variation"),
            when(col(CudaGpuTrace::NAME).eq("Kernel A"))
                .then(col("Kernel B"))
                .otherwise(col("Kernel A"))
                .alias("Opposing Kernel"),
        ])
        .with_column(
            (((lit(1.0) + col("Norm 99%")).pow(lit(0.75))
                * (lit(1.0) + col("Norm Coefficient of Variation")).pow(0.25))
                - lit(1.0))
            .alias("Interference Impact"),
        )
        .select([
            col(CudaGpuTrace::NAME),
            col("Opposing Kernel"),
            col("Norm 99%"),
            col("Norm Coefficient of Variation"),
            col("Interference Impact"),
        ])
        .sort(
            ["Interference Impact"],
            SortMultipleOptions::default().with_order_descending(true),
        );
    println!(
        "Paired Table: {:?}",
        paired_kernel_duration_summary.clone().collect().unwrap()
    );
    let tmp_aggression_score = paired_kernel_duration_summary
        .clone()
        .group_by(["Opposing Kernel"])
        .agg([col("Interference Impact").mean().alias("Aggression")])
        .sort(
            ["Aggression"],
            SortMultipleOptions::default().with_order_descending(true),
        );
    let global_aggression_mean = tmp_aggression_score
        .clone()
        .collect()
        .unwrap()
        .column("Aggression")
        .unwrap()
        .mean_reduce()
        .unwrap();
    let sensitivity_scores = paired_kernel_duration_summary
        .clone()
        .join(
            tmp_aggression_score.clone(),
            [col("Opposing Kernel")],
            [col("Opposing Kernel")],
            JoinArgs::new(JoinType::Inner),
        )
        .with_column(
            (col("Interference Impact")
                * (lit(global_aggression_mean) / (col("Aggression") + lit(1e-9))))
            .alias("Weighted Sensitivity"),
        );
    let sensitivity_score_summary = sensitivity_scores
        .clone()
        .group_by([CudaGpuTrace::NAME])
        .agg([
            col("Interference Impact").mean().alias("Naive Sensitivity"),
            col("Weighted Sensitivity")
                .mean()
                .alias("Weighted Sensitivity"),
        ]);
    let global_sensitivity_mean = sensitivity_score_summary
        .clone()
        .collect()
        .unwrap()
        .column("Naive Sensitivity")
        .unwrap()
        .mean_reduce()
        .unwrap();
    let aggression_scores = paired_kernel_duration_summary
        .clone()
        .join(
            sensitivity_score_summary.clone(),
            [col(CudaGpuTrace::NAME)],
            [col(CudaGpuTrace::NAME)],
            JoinArgs::new(JoinType::Inner),
        )
        .with_column(
            (col("Interference Impact")
                * (lit(global_sensitivity_mean) / (col("Naive Sensitivity") + lit(1e-9))))
            .alias("Weighted Aggression"),
        );
    let aggression_score_summary = aggression_scores
        .clone()
        .group_by(["Opposing Kernel"])
        .agg([
            col("Interference Impact").mean().alias("Naive Aggression"),
            col("Weighted Aggression")
                .mean()
                .alias("Weighted Aggression"),
        ]);
    let final_kernel_scorings = sensitivity_score_summary.clone().join(
        aggression_score_summary.clone(),
        [col(CudaGpuTrace::NAME)],
        [col("Opposing Kernel")],
        JoinArgs::new(JoinType::Inner),
    );
    let full_ncu_profiling_pairs = concat(pair_ncu_profile_rows, UnionArgs::default())
        .unwrap()
        .rename(["Kernel Name"], [CudaGpuTrace::NAME], true);
    let start = std::time::Instant::now();
    let [
        mut pivoted_naive_impact_scores,
        mut pivoted_weighted_sensitivity_scores,
        mut pivoted_weighted_aggression_scores,
        mut final_kernel_scorings,
        mut pivoted_mean_system_latency_summary,
        mut pivoted_p99_system_latency_summary,
        mut pivoted_contested_resources,
        mut pivoted_most_contested_resource,
    ] = collect_all_array([
        get_pivoted_table_for_attribute(
            sensitivity_scores.clone(),
            "Interference Impact",
            r"Victim \ Aggressor",
        ),
        get_pivoted_table_for_attribute(
            sensitivity_scores.clone(),
            "Weighted Sensitivity",
            r"Victim \ Aggressor",
        ),
        get_pivoted_table_for_attribute(
            aggression_scores.clone(),
            "Weighted Aggression",
            r"Victim \ Aggressor",
        ),
        final_kernel_scorings
            .clone()
            .sort([CudaGpuTrace::NAME], SortMultipleOptions::default()),
        pivoted_mean_system_latency_summary.clone(),
        pivoted_p99_system_latency_summary.clone(),
        get_pivoted_table_for_attribute(
            full_ncu_profiling_pairs
                .clone()
                .with_column(col("Contested Resources").list().join(lit(";"), true)),
            "Contested Resources",
            "Name",
        ),
        get_pivoted_table_for_attribute(
            full_ncu_profiling_pairs.clone(),
            "Main Bottleneck",
            "Name",
        ),
    ])
    .unwrap();
    let end = std::time::Instant::now();
    println!("Kernel Scores: {}", final_kernel_scorings);
    println!("Pivoted Naive Impacts: {}", pivoted_naive_impact_scores);
    println!(
        "Pivoted Weighted Sensitivity: {}",
        pivoted_weighted_sensitivity_scores
    );
    println!(
        "Pivoted Weighted Aggression: {}",
        pivoted_weighted_aggression_scores
    );
    println!("Contested Resources: {}", pivoted_contested_resources);
    println!("Main Bottlenecks: {}", pivoted_most_contested_resource);
    debug!("Pivoting took {:#?}", end - start);

    write_to_csv(
        &global_config.new_output_file("naive_impact_scores.csv"),
        &mut pivoted_naive_impact_scores,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("weighted_sensitivity_scores.csv"),
        &mut pivoted_weighted_sensitivity_scores,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("weighted_aggression_scores.csv"),
        &mut pivoted_weighted_aggression_scores,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("kernel_score_summary.csv"),
        &mut final_kernel_scorings,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("concurrency_efficiency_of_means.csv"),
        &mut pivoted_mean_system_latency_summary,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("concurrency_efficiency_of_p99.csv"),
        &mut pivoted_p99_system_latency_summary,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("contested_resources.csv"),
        &mut pivoted_contested_resources,
    )
    .unwrap();
    write_to_csv(
        &global_config.new_output_file("main_bottlenecks.csv"),
        &mut pivoted_most_contested_resource,
    )
    .unwrap();
}
