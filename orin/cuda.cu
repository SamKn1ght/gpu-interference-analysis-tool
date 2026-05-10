#include "generated_kernels.h"

// Synthetic Workloads

__global__ void strided_pythagorus(double* a, double* b, double* c, int size) {
    int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        c[i] = sqrt(a[i] * a[i] + b[i] + b[i]);
    }
}

__global__ void strided_reduce_sum(double* d, float* d_sum, int size) {
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
        atomicAdd(d_sum, (float)thread_sum[0]);
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
    data->n = (1 << 21); // ~ 16MB per array of doubles
    int bytes = data->n * sizeof(double);

    data->width = 2048;
    data->height = 2048;
    int image_pixels = data->width * data->height;
    int image_rgb_bytes = image_pixels * 3;

    data->input_width = 1024;
    data->input_height = 1024;
    data->kernel_width = 7;
    data->kernel_height = 7;
    int input_size = data->input_width * data->input_height;
    int input_bytes = input_size * sizeof(float);
    int kernel_size = data->kernel_width * data->kernel_height;
    int kernel_bytes = kernel_size * sizeof(float);
    int output_size = (data->input_width - data->kernel_width) * (data->input_height - data->kernel_height);
    int output_bytes = output_size * sizeof(float);

    data->batch_size = 4;
    data->grid_size = 32;
    data->num_faces = 12;
    data->num_vertices = 8;
    int total_voxels = data->batch_size * data->grid_size * data->grid_size * data->grid_size;
    int phi_size = total_voxels;
    int phi_bytes = total_voxels * sizeof(float);
    int faces_size = data->num_faces * 3;
    int faces_bytes = faces_size * sizeof(int32_t);
    int vertices_size = data->num_vertices * 3 * data->batch_size;
    int vertices_bytes = vertices_size * sizeof(float);

    cudaHostAlloc((void**) &data->h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_c, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_d_sum, sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_e, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_f, bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_g, bytes, cudaHostAllocDefault);

    cudaHostAlloc((void**) &data->h_rgb, image_rgb_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_gray, image_pixels, cudaHostAllocDefault);

    cudaHostAlloc((void**) &data->h_conv_input, input_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_conv_kernel, kernel_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_conv_output, output_bytes, cudaHostAllocDefault);

    cudaHostAlloc((void**) &data->h_phi, phi_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_faces, faces_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**) &data->h_vertices, vertices_bytes, cudaHostAllocDefault);

    cudaMalloc((void **)&data->d_a, bytes);
    cudaMalloc((void **)&data->d_b, bytes);
    cudaMalloc((void **)&data->d_c, bytes);
    cudaMalloc((void **)&data->d_d, bytes);
    cudaMalloc((void **)&data->d_d_sum, sizeof(float));
    cudaMalloc((void **)&data->d_e, bytes);
    cudaMalloc((void **)&data->d_f, bytes);
    cudaMalloc((void **)&data->d_g, bytes);

    cudaMalloc((void**) &data->d_rgb, image_rgb_bytes);
    cudaMalloc((void**) &data->d_gray, image_pixels);

    cudaMalloc((void**) &data->d_conv_input, input_bytes);
    cudaMalloc((void**) &data->d_conv_kernel, kernel_bytes);
    cudaMalloc((void**) &data->d_conv_output, output_bytes);

    cudaMalloc((void**) &data->d_phi, phi_bytes);
    cudaMalloc((void**) &data->d_faces, faces_bytes);
    cudaMalloc((void**) &data->d_vertices, vertices_bytes);

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
    for (int i = 0; i < input_size; i++) {
        data->h_conv_input[i] = (float)i;
    }
    for (int i = 0; i < kernel_size; i++) {
        data->h_conv_kernel[i] = (float)(10 * i);
    }
    for (int i = 0; i < output_size; i++) {
        data->h_conv_output[i] = (float)0;
    }

    float unit_square[data->num_vertices][3] = {
        {-0.5f, -0.5f, -0.5f},
        {0.5f, -0.5f, -0.5f},
        {0.5f,  0.5f, -0.5f},
        {-0.5f,  0.5f, -0.5f},
        {-0.5f, -0.5f,  0.5f},
        {0.5f, -0.5f,  0.5f},
        {0.5f,  0.5f,  0.5f},
        {-0.5f,  0.5f,  0.5f}
    };
    int32_t square_faces[data->num_faces][3] = {
        {0, 1, 2},
        {0, 2, 3},
        {4, 5, 6},
        {4, 6, 7},
        {0, 1, 5},
        {0, 5, 4},
        {2, 3, 7},
        {2, 7, 6},
        {0, 4, 7},
        {0, 7, 3},
        {1, 5, 6},
        {1, 6, 2},
    };
    for (int b = 0; b < data->batch_size; b++) {
        for (int i = 0; i < data->num_vertices * 3; i++) {
            data->h_vertices[b * data->num_vertices * 3 + i * 3 + 0] = unit_square[i][0];
            data->h_vertices[b * data->num_vertices * 3 + i * 3 + 1] = unit_square[i][1];
            data->h_vertices[b * data->num_vertices * 3 + i * 3 + 2] = unit_square[i][2];
        }
    }
    for (int i = 0; i < phi_size; i++) {
        data->h_phi[i] = 0.0;
    }

    cudaMemcpyAsync(data->d_a, data->h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_b, data->h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_c, data->h_c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_d, data->h_d, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_d_sum, data->h_d_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_e, data->h_e, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_f, data->h_f, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_g, data->h_g, bytes, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(data->d_rgb, data->h_rgb, image_rgb_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_gray, data->h_gray, image_pixels, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(data->d_conv_input, data->h_conv_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_conv_kernel, data->h_conv_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_conv_output, data->h_conv_output, output_bytes, cudaMemcpyHostToDevice);

    cudaMemcpyAsync(data->d_phi, data->h_phi, phi_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_vertices, data->h_vertices, vertices_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(data->d_faces, data->h_faces, faces_bytes, cudaMemcpyHostToDevice);
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

    cudaFreeHost(data->h_phi);
    cudaFreeHost(data->h_vertices);
    cudaFreeHost(data->h_faces);

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

    cudaFree(data->d_phi);
    cudaFree(data->d_vertices);
    cudaFree(data->d_faces);
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

// Signed Distance Field kernel authored by penincillin
// Repository: https://github.com/penincillin/SDF_ihmr/tree/master
// Licensed under MIT license
// This attribution applies to all code below this point until the next attribution
static __inline__ __device__ float dot(const float* x, const float* y) {
    float l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * y[i];
    }
    return l;
}

static __inline__ __device__ float dist(const float* x, const float* y) {
    float l = 0;
    float diff;
    for (int i=0; i<3; ++i) {
        diff = x[i] - y[i];
        l += diff * diff;
    }
    return sqrt(l);
}

static __inline__ __device__ float mag2(const float* x) {
    float l = 0;
    for (int i=0; i<3; ++i) {
        l += x[i] * x[i];
    }
    return l;
}



static __inline__ __device__ float point_segment_distance(const float* x0, const float* x1, const float* x2, float* r)
{
    float dx[3] = {x2[0]-x1[0], x2[1]-x1[1], x2[2]-x1[2]};
    float m2 = mag2(dx);
    // find parameter value of closest point on segment
    // float s12= (float) (dot(x2-x0, dx)/m2);
    float s12 = (float) (dot(x2, dx) - dot(x0, dx)) / m2;
    if (s12 < 0){
       s12 = 0;
    }
    else if (s12 > 1){
       s12 = 1;
    }
    for (int i=0; i < 3; ++i) {
        r[i] = s12*x1[i] + (1-s12) * x2[i];
    }
    // and find the distance
    return dist(x0, r);
}
static __inline__ __device__ float point_triangle_distance(const float* x0, const float* x1, const float* x2, const float* x3, float* r) {
   // first find barycentric coordinates of closest point on infinite plane
    float x13[3];
    float x23[3];
    float x03[3];
    for (int i=0; i<3; ++i) {
        x13[i] = x1[i] - x3[i];
        x23[i] = x2[i] - x3[i];
        x03[i] = x0[i] - x3[i];
    }
    float m13 = mag2(x13);
    float m23 = mag2(x23);
    float m33 = mag2(x03);
    float d = dot(x13, x23);
    float invdet=1.f/max(m13*m23-d*d,1e-30f);
    float a = dot(x13, x03);
    float b = dot(x23, x03);
    // the barycentric coordinates themselves
    float w23=invdet*(m23*a-d*b);
    float w31=invdet*(m13*b-d*a);
    float w12=1-w23-w31;

    if (w23>=0 && w31>=0 && w12>=0){ // if we're inside the triangle
        for (int i=0; i<3; ++i) {
            r[i] = w23*x1[i] + w31*x2[i]+w12*x3[i];
        }
        return dist(x0, r); 
    }
    else { // we have to clamp to one of the edges
        float r1[3] = {0,0,0};
        float r2[3] = {0,0,0};
        if (w23 > 0) {// this rules out edge 2-3 for us
            float d1 = point_segment_distance(x0,x1,x2,r1);
            float d2 = point_segment_distance(x0,x1,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else if (w31 > 0) {// this rules out edge 1-3
            float d1 = point_segment_distance(x0,x1,x2,r1);
            float d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
        else {// w12 must be >0, ruling out edge 1-2
            float d1 = point_segment_distance(x0,x1,x3,r1);
            float d2 = point_segment_distance(x0,x2,x3,r2);
            if (d1 < d2) {
                for (int i=0; i < 3; ++i) {
                    r[i] = r1[i];
                }
                return d1;
            }
            else {
                for (int i=0; i < 3; ++i) {
                    r[i] = r2[i];
                }
                return d2;
            }
        }
    }
}

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2];

static __inline__ __device__ int intersect_triangle(
               const float* orig, const float* dir,
		       const float* vert0, const float* vert1,
               const float* vert2, float* t, float *u, float *v) {

    float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
    float det,inv_det;
    
    /* find vectors for two edges sharing vert0 */
    SUB(edge1, vert1, vert0);
    SUB(edge2, vert2, vert0);
    
    /* begin calculating determinant - also used to calculate U parameter */
    CROSS(pvec, dir, edge2);
    
    /* if determinant is near zero, ray lies in plane of triangle */
    det = DOT(edge1, pvec);
    
    if (det > -EPSILON && det < EPSILON)
        return 0;
    inv_det = 1.0 / det;
    
    /* calculate distance from vert0 to ray origin */
    SUB(tvec, orig, vert0);
    
    /* calculate U parameter and test bounds */
    *u = DOT(tvec, pvec) * inv_det;
    if (*u < 0.0 || *u > 1.0)
        return 0;
    
    /* prepare to test V parameter */
    CROSS(qvec, tvec, edge1);
    
    /* calculate V parameter and test bounds */
    *v = DOT(dir, qvec) * inv_det;
    if (*v < 0.0 || (*u + *v) > 1.0)
        return 0;
    
    /* calculate t, ray intersects triangle */
    *t = DOT(edge2, qvec) * inv_det;

    
    return 1;
}

static __inline__ __device__ int triangle_ray_intersection(const float* origin, const float* dest,
    const float* v1, const float* v2, const float* v3, float* t) {

    float _dir[3] = {dest[0] - origin[0], dest[1] - origin[1], dest[2] - origin[2]};

    // t is the distance, u and v are barycentric coordinates
    // http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/code/raytri_tam.pdf
    float u, v;
    return intersect_triangle(origin, _dir, v1, v2, v3, t, &u, &v);
}

__global__ void sdf_cuda_kernel(
        float* phi,
        int32_t* faces,
        float* vertices,
        int batch_size,
        int num_faces,
        int num_vertices,
        int grid_size) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * grid_size * grid_size * grid_size) {
        return;
    }
    const int i = tid % grid_size;
    const int j = (tid / grid_size) % grid_size;
    const int k = (tid / (grid_size*grid_size)) % grid_size;
    const int bn = tid / (grid_size*grid_size*grid_size);
    const float dx = 2./(grid_size-1);
    const float center_x = -1 + (i + 0.5) * dx;
    const float center_y = -1 + (j + 0.5) * dx;
    const float center_z = -1 + (k + 0.5) * dx;

    const float center[3] = {center_x, center_y, center_z};
    int num_intersect = 0;
    float min_distance=1000;
    for (int f = 0; f < num_faces; ++f) {
        const int32_t* face = &faces[3*f];
        const int v1i = face[0];
        const int v2i = face[1];
        const int v3i = face[2];
        const float* v1 = &vertices[bn*num_vertices*3 + v1i*3];
        const float* v2 = &vertices[bn*num_vertices*3 + v2i*3];
        const float* v3 = &vertices[bn*num_vertices*3 + v3i*3];
        float closest_point[3];
        point_triangle_distance(center, v1, v2, v3, closest_point);
        float distance = dist(center, closest_point);

        if (distance < min_distance) {
            min_distance = distance;
        }

        float origin[3] = {-1.0, -1.0, -1.0};
        bool intersect = triangle_ray_intersection(center, origin, v1, v2, v3, &distance);

        if (intersect && distance >= 0) {
            num_intersect++;
        }
    }
    if (num_intersect % 2 == 0) {
        min_distance = 0.;
    }
    // if (num_intersect % 2 == 1) {
    //     min_distance *= -1;
    // }
    // phi[tid] = (float) num_intersect;
    // phi[bn*grid_size*grid_size*grid_size + k*grid_size*grid_size + j*grid_size + i] = min_distance;
    phi[tid] = min_distance;

    // if (num_intersect % 2 == 0) {
    //     phi[tid] = 0;
    // }
}
