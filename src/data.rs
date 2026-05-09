use std::{error::Error, fs::File, path::Path};

use polars::prelude::*;

pub fn collect_all_array<const N: usize>(
    lazy_frames: [LazyFrame; N],
) -> PolarsResult<[DataFrame; N]> {
    LazyFrame::collect_all_with_engine(
        lazy_frames.map(|f| f.logical_plan).to_vec(),
        Engine::Auto,
        OptFlags::default(),
    )?
    .try_into()
    .map_err(|_| {
        PolarsError::AssertionError(
            "Number of lazy frames mismatched to constant value N in collect_all_arr".into(),
        )
    })
}

pub fn write_to_csv(path: &Path, df: &mut DataFrame) -> PolarsResult<()> {
    let file = File::create(path).unwrap();
    CsvWriter::new(file).include_header(true).finish(df)
}

pub fn get_pivoted_table_for_attribute(
    frame: LazyFrame,
    attribute: &'static str,
    name_column_alias: &'static str,
) -> LazyFrame {
    frame
        .clone()
        .pivot(
            Selector::ByName {
                names: Arc::new([PlSmallStr::from_static("Opposing Kernel")]),
                strict: true,
            },
            Arc::new(
                frame
                    .select([col("Opposing Kernel")])
                    .unique(None, UniqueKeepStrategy::First)
                    .sort(
                        [PlSmallStr::from_static("Opposing Kernel")],
                        SortMultipleOptions::default(),
                    )
                    .collect()
                    .unwrap(),
            ),
            Selector::ByName {
                names: Arc::new([PlSmallStr::from_static(CudaGpuTrace::NAME)]),
                strict: true,
            },
            Selector::ByName {
                names: Arc::new([PlSmallStr::from_static(attribute)]),
                strict: true,
            },
            element().first(),
            true,
            PlSmallStr::from_static(""),
        )
        .sort(
            [PlSmallStr::from_static(CudaGpuTrace::NAME)],
            SortMultipleOptions::default(),
        )
        .rename([CudaGpuTrace::NAME], [name_column_alias], true)
}

pub fn pivot_ncu_data(frame: LazyFrame) -> LazyFrame {
    frame.clone().pivot(
        Selector::ByName {
            names: Arc::new([PlSmallStr::from_static(NcuData::METRIC_NAME)]),
            strict: true,
        },
        Arc::new(
            frame
                .select([col(NcuData::METRIC_NAME)])
                .sort(
                    [PlSmallStr::from_static(NcuData::METRIC_NAME)],
                    SortMultipleOptions::default(),
                )
                .collect()
                .unwrap(),
        ),
        Selector::ByName {
            names: Arc::new([PlSmallStr::from_static(NcuData::KERNEL_NAME)]),
            strict: true,
        },
        Selector::ByName {
            names: Arc::new([PlSmallStr::from_static(NcuData::METRIC_VALUE)]),
            strict: true,
        },
        element().first(),
        true,
        PlSmallStr::from_static(""),
    )
}

pub fn get_gpu_duration_summary(frame: LazyFrame) -> LazyFrame {
    frame
        .clone()
        .group_by([col(CudaGpuTrace::NAME)])
        .agg([
            col(CudaGpuTrace::DURATION).mean().alias("Mean"),
            col(CudaGpuTrace::DURATION).median().alias("Median"),
            col(CudaGpuTrace::DURATION).std(1).alias("Std. Dev"),
            col(CudaGpuTrace::DURATION)
                .quantile(lit(0.95), QuantileMethod::Linear)
                .alias("95%"),
            col(CudaGpuTrace::DURATION)
                .quantile(lit(0.99), QuantileMethod::Linear)
                .alias("99%"),
            col(CudaGpuTrace::DURATION).max().alias("Max"),
        ])
        .with_columns([
            (col("Mean") - col("Median")).alias("Skew"),
            (col("Std. Dev").cast(DataType::Float64) / col("Mean").cast(DataType::Float64))
                .alias("Coefficient of Variation"),
        ])
}

pub fn get_system_latency_summary(frame: LazyFrame) -> LazyFrame {
    frame
        .clone()
        .group_by([col("Kernel Name")])
        .agg([
            col("Missed Deadline").sum().alias("Missed Deadlines"),
            col("System Latency").mean().alias("System Latency Mean"),
            col("System Latency")
                .median()
                .alias("System Latency Median"),
            col("System Latency")
                .std(1)
                .alias("System Latency Std. Dev"),
            col("System Latency")
                .quantile(lit(0.99), QuantileMethod::Linear)
                .alias("System Latency 99%"),
            col("System Latency").max().alias("System Latency Max"),
        ])
        .rename(["Kernel Name"], [CudaApiTrace::NAME], true)
}

pub fn lazy_load_api_trace_dataframe(path: &Path) -> Result<LazyFrame, Box<dyn Error>> {
    Ok(LazyCsvReader::new(PlRefPath::try_from_path(path)?)
        .with_schema(Some(Arc::new(CudaApiTrace::get_schema())))
        .finish()?
        .with_columns([
            col(CudaApiTrace::START).cast(DataType::Duration(TimeUnit::Nanoseconds)),
            col(CudaApiTrace::DURATION).cast(DataType::Duration(TimeUnit::Nanoseconds)),
        ])
        .with_columns([
            (col(CudaApiTrace::START) + col(CudaApiTrace::DURATION)).alias(CudaApiTrace::END)
        ]))
}

pub fn lazy_load_gpu_trace_dataframe(path: &Path) -> Result<LazyFrame, Box<dyn Error>> {
    Ok(LazyCsvReader::new(PlRefPath::try_from_path(path)?)
        .with_schema(Some(Arc::new(CudaGpuTrace::get_schema())))
        .finish()?
        .with_columns([
            col(CudaApiTrace::START).cast(DataType::Duration(TimeUnit::Nanoseconds)),
            col(CudaApiTrace::DURATION).cast(DataType::Duration(TimeUnit::Nanoseconds)),
        ])
        .with_columns([
            (col(CudaGpuTrace::START) + col(CudaGpuTrace::DURATION)).alias(CudaGpuTrace::END)
        ]))
}

pub fn lazy_load_ncu_dataframe(path: &Path) -> Result<LazyFrame, Box<dyn Error>> {
    let selected_metrics = Series::new(
        PlSmallStr::from_static("Selected Metrics"),
        vec![
            "Compute (SM) Throughput",
            "L1/TEX Cache Throughput",
            "L2 Cache Throughput",
            "Mem Busy",
            "Max Bandwidth",
            "Achieved Occupancy",
        ],
    );
    Ok(LazyCsvReader::new(PlRefPath::try_from_path(path)?)
        .with_schema(Some(Arc::new(NcuData::get_schema())))
        .finish()?
        .filter(col(NcuData::METRIC_NAME).is_in(lit(selected_metrics).implode(), true))
        .filter(col(NcuData::METRIC_UNIT).eq(lit("%")))
        .select([
            col(NcuData::KERNEL_NAME),
            col(NcuData::METRIC_NAME),
            col(NcuData::METRIC_UNIT),
            col(NcuData::METRIC_VALUE),
        ])
        .with_column(col(NcuData::METRIC_VALUE).cast(DataType::Float64)))
}

trait ToSchema {
    fn get_schema() -> Schema;
}

pub struct CudaApiTrace;
impl CudaApiTrace {
    pub const START: &str = "Start (ns)";
    pub const END: &str = "End (ns)";
    pub const DURATION: &str = "Duration (ns)";
    pub const NAME: &str = "Name";
    pub const RESULT: &str = "Result";
    pub const CORR_ID: &str = "CorrID";
    pub const PID: &str = "Pid";
    pub const TID: &str = "Tid";
    pub const T_PRIO: &str = "T-Pri";
    pub const THREAD_NAME: &str = "Thread Name";
}
impl ToSchema for CudaApiTrace {
    fn get_schema() -> Schema {
        Schema::from_iter(vec![
            Field::new(Self::START.into(), DataType::Int64),
            Field::new(Self::DURATION.into(), DataType::Int64),
            Field::new(Self::NAME.into(), DataType::String),
            Field::new(Self::RESULT.into(), DataType::Int32),
            Field::new(Self::CORR_ID.into(), DataType::UInt64),
            Field::new(Self::PID.into(), DataType::Int32),
            Field::new(Self::TID.into(), DataType::Int32),
            Field::new(Self::T_PRIO.into(), DataType::Int8),
            Field::new(Self::THREAD_NAME.into(), DataType::String),
        ])
    }
}

pub struct CudaGpuTrace;
impl CudaGpuTrace {
    pub const START: &str = "Start (ns)";
    pub const END: &str = "End (ns)";
    pub const DURATION: &str = "Duration (ns)";
    pub const CORR_ID: &str = "CorrID";
    pub const GRID_X: &str = "GrdX";
    pub const GRID_Y: &str = "GrdY";
    pub const GRID_Z: &str = "GrdZ";
    pub const BLOCK_X: &str = "BlkX";
    pub const BLOCK_Y: &str = "BlkY";
    pub const BLOCK_Z: &str = "BlkZ";
    pub const REG_PER_THREAD: &str = "Reg/Trd";
    pub const STATIC_SHARED_MEM: &str = "StcSMem (MB)";
    pub const DYN_SHARED_MEM: &str = "DynSMem (MB)";
    pub const BYTES: &str = "Bytes (MB)";
    pub const THROUGHPUT: &str = "Throughput (MB/s)";
    pub const SOURCE_MEM_KIND: &str = "SrcMemKd";
    pub const DEST_MEM_KIND: &str = "DstMemKd";
    pub const DEVICE: &str = "Device";
    pub const CONTEXT: &str = "Ctx";
    pub const GREEN_CONTEXT: &str = "GreenCtx";
    pub const STREAM: &str = "Strm";
    pub const NAME: &str = "Name";
}
impl ToSchema for CudaGpuTrace {
    fn get_schema() -> Schema {
        Schema::from_iter(vec![
            Field::new(Self::START.into(), DataType::Int64),
            Field::new(Self::DURATION.into(), DataType::Int64),
            Field::new(Self::CORR_ID.into(), DataType::UInt64),
            Field::new(Self::GRID_X.into(), DataType::UInt32),
            Field::new(Self::GRID_Y.into(), DataType::UInt32),
            Field::new(Self::GRID_Z.into(), DataType::UInt32),
            Field::new(Self::BLOCK_X.into(), DataType::UInt32),
            Field::new(Self::BLOCK_Y.into(), DataType::UInt32),
            Field::new(Self::BLOCK_Z.into(), DataType::UInt32),
            Field::new(Self::REG_PER_THREAD.into(), DataType::UInt8), // MAX value of 255
            Field::new(Self::STATIC_SHARED_MEM.into(), DataType::Float32),
            Field::new(Self::DYN_SHARED_MEM.into(), DataType::Float32),
            Field::new(Self::BYTES.into(), DataType::Float32),
            Field::new(Self::THROUGHPUT.into(), DataType::Float32),
            Field::new(Self::SOURCE_MEM_KIND.into(), DataType::String),
            Field::new(Self::DEST_MEM_KIND.into(), DataType::String),
            Field::new(Self::DEVICE.into(), DataType::String),
            Field::new(Self::CONTEXT.into(), DataType::UInt32),
            Field::new(Self::GREEN_CONTEXT.into(), DataType::UInt32),
            Field::new(Self::STREAM.into(), DataType::UInt32),
            Field::new(Self::NAME.into(), DataType::String),
        ])
    }
}

pub struct NcuData;
impl NcuData {
    pub const ID: &str = "ID";
    pub const PID: &str = "Process ID";
    pub const PROCESS_NAME: &str = "Process Name";
    pub const HOST_NAME: &str = "Host Name";
    pub const KERNEL_NAME: &str = "Kernel Name";
    pub const CONTEXT: &str = "Context";
    pub const STREAM: &str = "Stream";
    pub const BLOCK_SIZE: &str = "Block Size";
    pub const GRID_SIZE: &str = "Grid Size";
    pub const DEVICE: &str = "Device";
    pub const COMPUTE_CAPACITY: &str = "Compute Capacity";
    pub const SECTION_NAME: &str = "Section Name";
    pub const METRIC_NAME: &str = "Metric Name";
    pub const METRIC_UNIT: &str = "Metric Unit";
    pub const METRIC_VALUE: &str = "Metric Value";
}
impl ToSchema for NcuData {
    fn get_schema() -> Schema {
        Schema::from_iter(vec![
            Field::new(Self::ID.into(), DataType::Int32),
            Field::new(Self::PID.into(), DataType::Int32),
            Field::new(Self::PROCESS_NAME.into(), DataType::String),
            Field::new(Self::HOST_NAME.into(), DataType::String),
            Field::new(Self::KERNEL_NAME.into(), DataType::String),
            Field::new(Self::CONTEXT.into(), DataType::UInt32),
            Field::new(Self::STREAM.into(), DataType::UInt32),
            Field::new(Self::BLOCK_SIZE.into(), DataType::String),
            Field::new(Self::GRID_SIZE.into(), DataType::String),
            Field::new(Self::DEVICE.into(), DataType::String),
            Field::new(Self::COMPUTE_CAPACITY.into(), DataType::String),
            Field::new(Self::SECTION_NAME.into(), DataType::String),
            Field::new(Self::METRIC_NAME.into(), DataType::String),
            Field::new(Self::METRIC_UNIT.into(), DataType::String),
            Field::new(Self::METRIC_VALUE.into(), DataType::String),
        ])
    }
}
