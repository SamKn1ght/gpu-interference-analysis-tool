#[derive(serde::Deserialize, Debug)]
pub struct CudaConfig {
    pub kernels: Vec<Kernel>,
    pub setup: SetupFunction,
    #[serde(default)]
    pub free: FreeFunction,
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
