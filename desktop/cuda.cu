#include "generated_kernels.h"

// Synthetic Workloads

__global__ void strided_pythagorus(double* a, double* b, double* c, int size) {
    int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        c[i] = sqrt(a[i] * a[i] + b[i] + b[i]);
    }
}

__global__ void strided_reduce_sum(double* d, double* d_sum, int size) {
    extern __shared__ double thread_sum[]; // Expected threads per block * sizeof(double)

    int thread_id = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        local_sum += d[i];
    }

    thread_sum[thread_id] = local_sum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (thread_id < i) {
            thread_sum[i] += thread_sum[thread_id + i];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        atomicAdd(d_sum, thread_sum[0]);
    }
}

__global__ void cellular_vec_add(double* e, double* f, double* g, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        g[i] = e[i] + f[i];
    }
}

extern "C" {

void setup(DataPointers* data) {
    data->n = (1 << 24); // ~ 128MB per array of doubles
    int bytes = data->n * sizeof(double);

    data->width = 8192;
    data->height = 8192;
    int image_pixels = data->width * data->height;
    int image_rgb_bytes = image_pixels * 3;

    data->input_width = 4096;
    data->input_height = 4096;
    data->kernel_width = 7;
    data->kernel_height = 7;
    int input_size = data->input_width * data->input_height;
    int input_bytes = input_size * sizeof(float);
    int kernel_size = data->kernel_width * data->kernel_height;
    int kernel_bytes = kernel_size * sizeof(float);
    int output_size = (data->input_width - data->kernel_width) * (data->input_height - data->kernel_height);
    int output_bytes = output_size * sizeof(float);

    cudaHostAlloc((void**) &data->h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_c, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d_sum, sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_e, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_f, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_g, bytes, cudaHostAllocDefault);

    cudaHostAlloc((void**) &data->h_rgb, image_rgb_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_gray, image_pixels, cudaHostAllocDefault);

    cudaHostAlloc((void**) &data->h_conv_input, input_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_conv_kernel, kernel_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_conv_output, output_bytes, cudaHostAllocDefault);

    cudaMalloc((void **)&data->d_a, bytes);
    cudaMalloc((void **)&data->d_b, bytes);
    cudaMalloc((void **)&data->d_c, bytes);
    cudaMalloc((void **)&data->d_d, bytes);
    cudaMalloc((void **)&data->d_d_sum, sizeof(double));
    cudaMalloc((void **)&data->d_e, bytes);
    cudaMalloc((void **)&data->d_f, bytes);
    cudaMalloc((void **)&data->d_g, bytes);

    cudaMalloc((void**) &data->d_rgb, image_rgb_bytes);
    cudaMalloc((void**) &data->d_gray, image_pixels);

    cudaMalloc((void**) &data->d_conv_input, input_bytes);
    cudaMalloc((void**) &data->d_conv_kernel, kernel_bytes);
    cudaMalloc((void**) &data->d_conv_output, output_bytes);

    for (int i = 0; i < data->n; i++) {
        data->h_a[i] = (double)(i + 1);
        data->h_b[i] = (double)(data->n - i);
        data->h_c[i] = 0.0;
        data->h_d[i] = 1.0;
        data->h_e[i] = (double)(2 * i);
        data->h_f[i] = (double)(3 * i);
        data->h_g[i] = 0.0;
    }
    data->h_d_sum[0] = 0.0;
    for (int i = 0; i < image_pixels; i++) {
        data->h_rgb[i * 3] = (unsigned char)(i % (256 * 256));
        data->h_rgb[i * 3 + 1] = (unsigned char)(i % 256);
        data->h_rgb[i * 3 + 2] = (unsigned char)i;
        data->h_gray[i] = (unsigned char)0;
    }
    // for (int i = 0; i < input_size; i++) {
    //     data->h_conv_input[i] = (float)i;
    // }
    // for (int i = 0; i < kernel_size; i++) {
    //     data->h_conv_kernel[i] = (float)(10 * i);
    // }
    // for (int i = 0; i < output_size; i++) {
    //     data->h_conv_output[i] = (float)0;
    // }

    cudaMemcpyAsync(data->d_a, data->h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_b, data->h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_c, data->h_c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_d, data->h_d, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_d_sum, data->h_d_sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_e, data->h_e, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_f, data->h_f, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_g, data->h_g, bytes, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(data->d_rgb, data->h_rgb, image_rgb_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_gray, data->h_gray, image_pixels, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(data->d_conv_input, data->h_conv_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_conv_kernel, data->h_conv_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_conv_output, data->h_conv_output, output_bytes, cudaMemcpyHostToDevice);
}

void free_data(DataPointers* data) {
    cudaFreeHost(data->h_a);
    cudaFreeHost(data->h_b);
    cudaFreeHost(data->h_c);
    cudaFreeHost(data->h_d);
    cudaFreeHost(data->h_d_sum);
    cudaFreeHost(data->h_e);
    cudaFreeHost(data->h_f);
    cudaFreeHost(data->h_g);

    cudaFreeHost(data->h_rgb);
    cudaFreeHost(data->h_gray);

    cudaFreeHost(data->h_conv_input);
    cudaFreeHost(data->h_conv_kernel);
    cudaFreeHost(data->h_conv_output);

    cudaFree(data->d_a);
    cudaFree(data->d_b);
    cudaFree(data->d_c);
    cudaFree(data->d_d);
    cudaFree(data->d_d_sum);
    cudaFree(data->d_e);
    cudaFree(data->d_f);
    cudaFree(data->d_g);

    cudaFree(data->d_rgb);
    cudaFree(data->d_gray);

    cudaFree(data->d_conv_input);
    cudaFree(data->d_conv_kernel);
    cudaFree(data->d_conv_output);
}

}

// Real Workloads

__global__ void grayscale_filter(unsigned char* rgb, unsigned char* gray, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;

    if (i < size) {
        int rgb_offset = i * 3;
        unsigned char r = rgb[rgb_offset];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];
        
        gray[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }

}

__global__ void convolution_2d(
    float* conv_input, 
    float* conv_kernel, 
    float* conv_output, 
    int input_width, int input_height, 
    int kernel_width, int kernel_height
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int output_width = input_width - kernel_width + 1;
    int output_height = input_height - kernel_height + 1;
    int total_outputs = output_width * output_height;

    if (thread_id < total_outputs) {
        int row = thread_id / output_width;
        int col = thread_id % output_width;

        float sum = 0.0f;

        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                int inputRow = row + i;
                int inputCol = col + j;
                
                sum += conv_input[inputRow * input_width + inputCol] * conv_kernel[i * kernel_width + j];
            }
        }
        
        conv_output[thread_id] = sum;
    }
}
