use std::collections::BTreeMap;

use crate::cuda::{CudaConfig, Kernel};

pub struct GpuPipelineView<'a> {
    pub name: &'a str,
    pub kernels: Vec<&'a Kernel>,
}
impl<'a> GpuPipelineView<'a> {
    pub fn collect_from_config(config: &'a CudaConfig) -> Vec<Self> {
        let kernel_map = config
            .kernels
            .iter()
            .map(|kernel| (kernel.name.as_str(), kernel))
            .collect::<BTreeMap<&str, &Kernel>>();
        let mut pipelines = Vec::with_capacity(config.pipelines.len());
        for pipeline in &config.pipelines {
            pipelines.push(Self {
                name: &pipeline.name,
                kernels: pipeline
                    .kernel_names
                    .iter()
                    .map(|name| {
                        *kernel_map
                            .get(name.as_str())
                            .expect("Kernel should exist at this point")
                    })
                    .collect::<Vec<_>>(),
            })
        }
        pipelines
    }
}
