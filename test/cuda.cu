#include "../generated/generated_kernels.h"

__global__ void vec_add(double* a, double* b, double* c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) { c[i] = a[i] + b[i]; }
}

__global__ void vec_mul(double* a, double* b, double* d, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) { d[i] = a[i] * b[i]; }
}

void setup(DataPointers* data) {

    data->n = 100000;
    int bytes = data->n * sizeof(double);

    cudaHostAlloc((void**) &data->h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_c, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d, bytes, cudaHostAllocDefault);
    cudaMalloc((void **)&data->d_a, bytes);
    cudaMalloc((void **)&data->d_b, bytes);
    cudaMalloc((void **)&data->d_c, bytes);
    cudaMalloc((void **)&data->d_d, bytes);

    for (int i = 0.0; i < data->n; i++) {
        data->h_a[i] = (double)(i + 1);
        data->h_b[i] = (double)(data->n - i);
        data->h_c[i] = 0.0;
        data->h_d[i] = 0.0;
    }

    cudaMemcpy(data->d_a, data->h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_b, data->h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_c, data->h_c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_d, data->h_d, bytes, cudaMemcpyHostToDevice);
}

void free_data(DataPointers* data) {
    cudaFreeHost(data->h_a);
    cudaFreeHost(data->h_b);
    cudaFreeHost(data->h_c);
    cudaFreeHost(data->h_d);
    cudaFree(data->d_a);
    cudaFree(data->d_b);
    cudaFree(data->d_c);
    cudaFree(data->d_d);
}
