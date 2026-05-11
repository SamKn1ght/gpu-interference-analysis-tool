# Gpu Interference Analysis Tool

This is a Rust tool built to automate the collection of timing and resource utilisation data to
allow for the automated suggestion of bottlenecks that could occur during concurrent execution of
Cuda kernels.

## Build

To build this project run:
```sh
cargo build
```

## Usage

This tool requires a user to define a yaml file containing the launch information for each kernel
as well as a code file that contains the implementations for each kernel.

The options within the yaml file are limited but a full example can be found in the `/desktop` or
`/orin` folders. The required fields within the kernels array are: name, blocks, threads, stream.

Args is interpreted to be empty if no arguments are passed. Args must be listed within the same
order as they would be placed within the launch call to ensure correct behaviour.

The data structure is automatically generated and is expected to be the only argument for the user
setup function. This will have pointers for the host and device side memory to be allocated for any
pointer types as well as having host only memory for non-pointer types. This type is derived from
the args to the setup function within the configuration file and will prefix host memory with `h_`
and device memory with `d_`.

Due to limitations with Compute profiling, it is disabled by default and can be enabled using the
`--full` flag. Additionally if sudo is needed to use ncu properly then the `--sudo` flag allows
for the command to be called using sudo for this session.
