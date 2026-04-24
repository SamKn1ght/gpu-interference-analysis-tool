#include "generated_kernels.h"

__global__ void vec_add(double* a, double* b, double* c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = sqrt(a[i] + b[i]);
    }
}

__global__ void vec_mul(double* a, double* b, double* d, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) { d[i] = a[i] * b[i]; }
}

__global__ void avg_h_blur(double* e, double* f, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 && i < size - 1) {
        f[i] = (e[i - 1] + e[i] + e[i + 1]) / 3.0;
    } else {
        f[i] = e[i];
    }
}

extern "C" {

void setup(DataPointers* data) {
    data->n = (1 << 27) - 1;
    int bytes = data->n * sizeof(double);

    cudaHostAlloc((void**) &data->h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_c, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_e, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_f, bytes, cudaHostAllocDefault);
    cudaMalloc((void **)&data->d_a, bytes);
    cudaMalloc((void **)&data->d_b, bytes);
    cudaMalloc((void **)&data->d_c, bytes);
    cudaMalloc((void **)&data->d_d, bytes);
    cudaMalloc((void **)&data->d_e, bytes);
    cudaMalloc((void **)&data->d_f, bytes);

    for (int i = 0; i < data->n; i++) {
        data->h_a[i] = (double)(i + 1);
        data->h_b[i] = (double)(data->n - i);
        data->h_c[i] = 0.0;
        data->h_d[i] = 0.0;
        data->h_e[i] = (double)(2 * i);
        data->h_f[i] = 0.0;
    }

    cudaMemcpyAsync(data->d_a, data->h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_b, data->h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_c, data->h_c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_d, data->h_d, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_e, data->h_e, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_f, data->h_f, bytes, cudaMemcpyHostToDevice);
}

void free_data(DataPointers* data) {
    cudaFreeHost(data->h_a);
    cudaFreeHost(data->h_b);
    cudaFreeHost(data->h_c);
    cudaFreeHost(data->h_d);
    cudaFreeHost(data->h_e);
    cudaFreeHost(data->h_f);
    cudaFree(data->d_a);
    cudaFree(data->d_b);
    cudaFree(data->d_c);
    cudaFree(data->d_d);
    cudaFree(data->d_e);
    cudaFree(data->d_f);
}

}
