use log::warn;

#[derive(serde::Deserialize, Debug)]
pub struct CudaConfig {
    pub kernels: Vec<Kernel>,
    pub setup: SetupFunction,
    #[serde(default)]
    pub free: FreeFunction,
    #[serde(default)]
    pub streams: Vec<Stream>,
}
impl CudaConfig {
    /// Validates the Config struct creating any implicitly defined structures
    ///
    /// This function must be called before use in templates to ensure validity of the structure
    ///
    /// Returns whether any warnings were emitted
    pub fn validate(&mut self) -> bool {
        let mut has_warning = false;
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
                has_warning = true;
            }
        }
        has_warning
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct Kernel {
    #[serde(default = "Kernel::default_return_type")]
    pub return_type: String,
    pub name: String,
    #[serde(default)]
    pub args: Vec<FunctionArg>,
    pub blocks: u32,
    pub threads: u32,
    pub stream: String,
}
impl Kernel {
    fn default_return_type() -> String {
        String::from("void")
    }
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

#[derive(serde::Deserialize, Debug)]
pub struct FunctionArg {
    pub name: String,
    #[serde(rename = "type")]
    pub datatype: String,
}
