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


void print_arr(int* data, size_t Nx, size_t Ny);

#define EPSILON (1e-3)
void cmp_body_arrs(int* seq, int* parallel, size_t n, size_t* bad_calcs, float* max_error) {
    *bad_calcs = 0;
	*max_error = 0;
	for (int i = 0; i < n; i++) {
		int error = abs(parallel[i] - seq[i]);
		if (error > EPSILON) {
			*max_error = error;
			(*bad_calcs)++;

			if (*bad_calcs > 3) continue;

			printf("Mismatch at index %d:\n", i);
			printf("  CUDA: (%d)\n", parallel[i]);
			printf("  CPU : (%d)\n", seq[i]);
			printf("  Diff: (%d)\n", error);
		}
  	}
}

__global__ void init_arr(int *data, int n, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, index, 0, &state);

    for (int i = index; i < n; i += stride) {
        data[i] = (int)(curand_uniform(&state) * INT_MAX) % MAX_VAL;
    }
}

__device__ int wrap_index(int index, int limit) {
    return (index + limit) % limit;  // Handles negative indices properly
}

__global__ void convolution(int* frame, int* end_frame, uint32_t Nx, uint32_t Ny,
                            uint32_t kernel_size, float* global_kernel, int start_y, int chunk_size) {

    extern __shared__ float shared_kernel[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    if (threadIdx.x < kernel_size * kernel_size) {
        shared_kernel[threadIdx.x] = global_kernel[threadIdx.x];
    }

    __syncthreads();

    for (int tid = index; tid < Nx * chunk_size; tid += stride) {
        int x = tid % Nx;
        int y = (tid / Nx) + start_y;
        if (y >= Ny) return;

        int sum = 0;
        int offset = kernel_size / 2;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int yi = wrap_index(y + i - offset, Ny);
                int xj = wrap_index(x + j - offset, Nx);

                sum += frame[yi * Nx + xj] * shared_kernel[i * kernel_size + j];
            }
        }
        end_frame[y * Nx + x] = sum;
    }
}




void get_window(int* arr, int x, int y, size_t Nx, size_t Ny, 
    int* window, size_t kernel_size);
int convolve(int* window, float* kernel, size_t kernel_size);

int main(const int argc, const char** argv) {
    size_t Nx = 1000;
    size_t Ny = 1000;
    if (argc == 2) {
        Nx = atoi(argv[1]);
    }
    else if (argc == 3) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
    }

    size_t mat_size = Nx * Ny;
    size_t bytes = mat_size * sizeof(int);
    
    int *frame, *end_frame;
    CHECK_CUDA(cudaMalloc(&frame, bytes));
    CHECK_CUDA(cudaMalloc(&end_frame, bytes));

    int deviceId;
	CHECK_CUDA(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
	int sm = prop.multiProcessorCount;
    int num_blocks = 2 * sm;
	int threads_per_block = 4 * MIN_CUDA_THREADS;

    init_arr<<<num_blocks, threads_per_block>>>(frame, mat_size, time(NULL));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    #ifdef DEBUG
    int *frame_seq, *end_frame_seq;
    frame_seq = (int*)malloc(bytes);
    end_frame_seq = (int*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(frame_seq, frame, bytes, cudaMemcpyDeviceToHost));
    
    #endif

    size_t kernel_size_bytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    float *kernel;
    CHECK_CUDA(cudaMallocManaged(&kernel, kernel_size_bytes));
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i ++) kernel[i] = 0.5f;

    size_t partition_size = (mat_size + num_blocks - 1) / num_blocks;
    cudaStream_t* streams = (cudaStream_t*)malloc(num_blocks * sizeof(cudaStream_t));   
    for (int i = 0; i < num_blocks; i++) {
        cudaStreamCreate(&streams[i]);
    }

    StartTimer();
    for (int i = 0; i < num_blocks; i++) {
        int start_ind = i * partition_size;

        convolution<<<num_blocks, threads_per_block, kernel_size_bytes>>>(frame, end_frame, Nx, Ny, KERNEL_SIZE, kernel, start_ind, partition_size);
        CHECK_CUDA(cudaGetLastError());
    }

    for (int i = 0; i < num_blocks; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    int* end_frame_h = (int*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(end_frame_h, end_frame, bytes, cudaMemcpyDeviceToHost));
    const double tElapsed = GetTimer() / 1000.0;

    #ifdef DEBUG
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            int window[KERNEL_SIZE * KERNEL_SIZE];
            get_window(frame_seq, j, i, Nx, Ny, window, KERNEL_SIZE);
            int val = convolve(window, kernel, KERNEL_SIZE);
            end_frame_seq[i * Nx + j] = convolve(window, kernel, KERNEL_SIZE);
        }
    }

    float max_error;
    size_t bad_calcs;
    cmp_body_arrs(end_frame_seq, end_frame_h, mat_size, &bad_calcs, &max_error);
	printf("Num of bad_calcs %zu with max %f \n", bad_calcs, max_error);
    #endif

    printf("%zu, %zu, %zu, %lf\n", Nx, Ny, mat_size, tElapsed);

    free(end_frame_h);
    CHECK_CUDA(cudaFree(frame));
    CHECK_CUDA(cudaFree(end_frame));

    for (int i = 0; i < num_blocks; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    #ifdef DEBUG
    free(frame_seq);
    free(end_frame_seq);
    #endif

    return 0;
}

void init_arr(int* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = rand() % MAX_VAL;
    }
}

void print_arr(int* data, size_t Nx, size_t Ny) {
    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            printf("%d, ", data[i * Nx + j]);
        }
        printf("\n");
    }
}


void get_window(int* arr, int x, int y, size_t Nx, size_t Ny, 
    int* window, size_t kernel_size) {
    
    int offset = kernel_size / 2;

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int yi = WRAP_INDEX(y + i - offset, Ny);  
            int xj = WRAP_INDEX(x + j - offset, Nx);  
            
            window[i * kernel_size + j] = arr[yi * Nx + xj];  
        }
    }
}


int convolve(int* window, float* kernel, size_t kernel_size) {
    int sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            size_t ind = i * kernel_size + j;
            sum += window[ind] * kernel[ind];
        }
    }
    return sum;
}
