use std::collections::BTreeMap;

use crate::cuda::{CudaConfig, Kernel, Stream};

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

pub struct PairedKernelView<'a> {
    pub kernels: [&'a Kernel; 2],
    pub streams: [&'a Stream; 2],
}
impl<'a> PairedKernelView<'a> {
    pub fn to_pair_name(&self) -> String {
        format!("{}-{}", self.kernels[0].name, self.kernels[1].name)
    }
    pub fn iter_unique_kernel_pairs(config: &'a CudaConfig) -> impl Iterator<Item = Self> {
        let stream_map = Self::generate_stream_lookup_map(config);
        Self::get_kernel_pairings(config)
            .filter(|(a, b)| a != b)
            .map(move |(a, b)| Self {
                kernels: [a, b],
                streams: [
                    stream_map
                        .get(a.stream.as_str())
                        .expect("Stream should exist at this point"),
                    stream_map
                        .get(b.stream.as_str())
                        .expect("Streams should exist at this point"),
                ],
            })
    }
    fn generate_stream_lookup_map(config: &'a CudaConfig) -> BTreeMap<&'a str, &'a Stream> {
        config
            .streams
            .iter()
            .map(|stream| (stream.name.as_str(), stream))
            .collect::<BTreeMap<&str, &Stream>>()
    }
    /// Gets all pairings of kernels within a CudaConfig struct
    fn get_kernel_pairings(
        config: &'a CudaConfig,
    ) -> impl Iterator<Item = (&'a Kernel, &'a Kernel)> {
        config
            .kernels
            .iter()
            .enumerate()
            .flat_map(|(i, a)| config.kernels.iter().skip(i + 1).map(move |b| (a, b)))
    }
}
