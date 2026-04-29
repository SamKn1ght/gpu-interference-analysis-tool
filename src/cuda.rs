use std::{collections::BTreeSet, time::Duration};

use log::{error, warn};

#[derive(serde::Deserialize, Debug)]
pub struct CudaConfig {
    pub kernels: Vec<Kernel>,
    #[serde(default)]
    pub pipelines: Vec<GpuPipeline>,
    pub setup: SetupFunction,
    #[serde(default)]
    pub free: FreeFunction,
    #[serde(default)]
    pub streams: Vec<Stream>,
    #[serde(default)]
    pub frame_budget: Option<Delay>,
    #[serde(default = "CudaConfig::default_iterations")]
    pub iterations: u32,
}
impl CudaConfig {
    /// Validates the Config struct creating any implicitly defined structures
    ///
    /// This function must be called before use in templates to ensure validity of the structure
    ///
    /// Returns whether any warnings were emitted
    pub fn validate(&mut self) -> bool {
        let kernel_set = BTreeSet::from_iter(self.kernels.iter().map(|kernel| &kernel.name));
        let mut has_error = false;
        for kernel in &self.kernels {
            if !self
                .streams
                .iter()
                .any(|stream| stream.name == kernel.stream)
            {
                self.streams.push(Stream {
                    name: kernel.stream.clone(),
                    priority: None,
                });
                warn!("Created implicit stream : {}", &kernel.stream);
            }
        }
        for pipeline in &self.pipelines {
            for kernel_name in &pipeline.kernel_names {
                if !kernel_set.contains(kernel_name) {
                    error!(
                        "Pipeline contains kernel ({}) not defined within the kernels section",
                        kernel_name
                    );
                    has_error = true;
                }
            }
        }
        has_error
    }
    fn default_iterations() -> u32 {
        50
    }
}

#[derive(serde::Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct Kernel {
    #[serde(default = "Kernel::default_return_type")]
    pub return_type: String,
    pub name: String,
    #[serde(default)]
    pub args: Vec<FunctionArg>,
    pub blocks: u32,
    pub threads: u32,
    #[serde(default = "Kernel::default_shared_memory_bytes")]
    pub shared_memory_bytes: u32,
    pub stream: String,
}
impl Kernel {
    fn default_return_type() -> String {
        String::from("void")
    }
    fn default_shared_memory_bytes() -> u32 {
        0
    }
    pub fn get_stream<'a>(&self, config: &'a CudaConfig) -> Option<&'a Stream> {
        config
            .streams
            .iter()
            .filter(|s| s.name == self.stream)
            .next()
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct GpuPipeline {
    pub name: String,
    #[serde(default)]
    pub kernel_names: Vec<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct Stream {
    pub name: String,
    #[serde(default)]
    pub priority: Option<i32>,
}

#[derive(serde::Deserialize, Debug)]
pub struct SetupFunction {
    #[serde(default = "SetupFunction::default_return_type")]
    pub return_type: String,
    #[serde(default = "SetupFunction::default_name")]
    pub name: String,
    #[serde(default)] // Default to no arguments
    pub args: Vec<FunctionArg>,
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
pub struct FreeFunction {
    #[serde(default = "FreeFunction::default_return_type")]
    pub return_type: String,
    #[serde(default = "FreeFunction::default_name")]
    pub name: String,
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

#[derive(serde::Deserialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct FunctionArg {
    pub name: String,
    #[serde(rename = "type")]
    pub datatype: String,
}

#[derive(serde::Deserialize, Debug)]
pub struct Delay {
    #[serde(with = "humantime_serde")]
    pub duration: Duration,
    pub method: DelayMethod,
}

#[derive(serde::Deserialize, Debug, PartialEq, Eq)]
pub enum DelayMethod {
    Busy,
    Sleep,
}
