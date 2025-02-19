#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../common/timer.h"
#include "../common/cuda_help.h"

#define MAX_VAL         (255)
#define KERNEL_SIZE     (3)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define WRAP_INDEX(idx, max) (((idx) + (max)) % (max))
// #define DEBUG

int num_blocks = 2;
int threads_per_block = 4 * MIN_CUDA_THREADS;

void print_arr(double* data, size_t N);

__global__ void init_arr(double *data_d, int n, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, index, 0, &state);

    for (int i = index; i < n; i += stride) {
        data_d[i] = (double)(curand_uniform(&state) * INT_MAX);
        // data_d[i] = i + 1;
    }
}

__global__ void neighbor_kernel(double* data_d, size_t n) {
    size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    size_t stride = 2 * (blockDim.x * gridDim.x);

    for (size_t i = index; i < n; i += stride) {
        if (i + 1 < n) {
            data_d[i / 2] = data_d[i] + data_d[i + 1];
        } else {
            data_d[i / 2] = data_d[i];  
        }
    }
}

double reduction_neighbor(double* data_d, size_t N) {
    while (N > 1) {
        size_t newN = (N + 1) / 2;
        neighbor_kernel<<<num_blocks, threads_per_block>>>(data_d, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        N = newN;
    }

    double val;
    CHECK_CUDA(cudaMemcpy(&val, data_d, sizeof(double), cudaMemcpyDeviceToHost));

    return val;
}

__global__ void stride_kernel(double* data_d, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t window_stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n; i += window_stride) {
        size_t stride = (n + 1) / 2;
        if (i + stride < n) {
            data_d[i] += data_d[i + stride];
        }
    }
}

double reduction_stride(double* data_d, size_t N) {
    while (N > 1) {
        size_t newN = (N + 1) / 2;
        stride_kernel<<<num_blocks, threads_per_block>>>(data_d, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        N = newN;
    }

    double val;
    CHECK_CUDA(cudaMemcpy(&val, data_d, sizeof(double), cudaMemcpyDeviceToHost));

    return val;
}

__global__ void seg_scan_red_kernel(double* data_d, size_t n, size_t stride) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t window_stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n; i += window_stride) {
        size_t target_idx = (i + 1) * (2 * stride) - 1;

        if (target_idx < n) { 
            data_d[target_idx] += data_d[target_idx - stride];
        }
    }
}

__global__ void seg_scan_ds_kernel(double* data_d, size_t n, size_t stride) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t window_stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n; i += window_stride) {
        size_t target_idx = (i + 1) * (2 * stride) - 1;

        if (target_idx < n) { 
            double tmp = data_d[target_idx - stride];
            data_d[target_idx - stride] = data_d[target_idx];
            data_d[target_idx] += tmp;
        }
    }
}


void seg_scan(double* data_d, size_t n, double* sum) {
    // pad the array
    size_t old_n = n;
    size_t bytes = n * sizeof(double);
    size_t old_bytes = bytes;
    double* arr_d;
    if ((n & (n - 1)) != 0) {
        n = 1 << ((size_t)log2(n) + 1);
        bytes = n * sizeof(bytes);
    }
    
    CHECK_CUDA(cudaMalloc(&arr_d, bytes));
    CHECK_CUDA(cudaMemcpy(arr_d, data_d, old_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(&arr_d[old_n], 0, bytes - old_bytes));

    for (size_t stride = 1; stride < n; stride *= 2) {
        seg_scan_red_kernel<<<num_blocks, threads_per_block>>>(arr_d, n, stride);
        CHECK_CUDA(cudaGetLastError());
        
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(sum, &arr_d[n - 1], sizeof(double), cudaMemcpyDeviceToHost));
    double zero = 0;
    CHECK_CUDA(cudaMemcpy(&arr_d[n - 1], &zero, sizeof(double), cudaMemcpyHostToDevice));

    for (size_t stride = n / 2; stride > 0; stride /= 2) {
        seg_scan_ds_kernel<<<num_blocks, threads_per_block>>>(arr_d, n, stride);
        CHECK_CUDA(cudaGetLastError());
        
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(data_d, arr_d, old_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(arr_d));
}



int main(const int argc, const char** argv) {
    size_t N = 10;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    size_t bytes = N * sizeof(double);
    double *arr_neighbors_d, *arr_stride_d, *arr_seg_scan_d;

    CHECK_CUDA(cudaMalloc(&arr_neighbors_d, bytes));
    CHECK_CUDA(cudaMalloc(&arr_stride_d, bytes));
    CHECK_CUDA(cudaMalloc(&arr_seg_scan_d, bytes));
    
    int deviceId;
	CHECK_CUDA(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
	int sm = prop.multiProcessorCount;
    num_blocks = 2 * sm;
    threads_per_block = 4 * MIN_CUDA_THREADS;
    
    init_arr<<<num_blocks, threads_per_block>>>(arr_neighbors_d, N, time(NULL));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(arr_stride_d, arr_neighbors_d, bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(arr_seg_scan_d, arr_neighbors_d, bytes, cudaMemcpyDeviceToDevice));

    StartTimer();
    
    double neighbor = reduction_neighbor(arr_neighbors_d, N);
    
    const double t_neighbor = GetTimer() / 1000.0;
    
    StartTimer();

    double stride = reduction_stride(arr_stride_d, N);

    const double t_stride = GetTimer() / 1000.0;

    StartTimer();

    double sum;
    seg_scan(arr_seg_scan_d, N, &sum);
    
    const double t_scan = GetTimer() / 1000.0;
    printf("%zu,%lf,%lf,%lf,%lf,%lf,%lf\n", N, 
        neighbor, t_neighbor, stride, t_stride, sum, t_scan);
 
    CHECK_CUDA(cudaFree(arr_neighbors_d));
    CHECK_CUDA(cudaFree(arr_stride_d));

    return 0;
}

void print_arr(double* data, size_t N) {
    for (int i = 0; i < N; i++) {
        printf("%lf, ", data[i]);
    }
    printf("\n");
}

