#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <curand_kernel.h>
#include <cuda_runtime.h>


using namespace std::chrono;
using namespace _V2;


#define CHECK_CUDA(call) \
        if ((call) != cudaSuccess) { \
                  fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
                  exit(EXIT_FAILURE); \
        }

__global__ void init_mat(float *data, int N, unsigned long seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, row, 0, &state);
    
    if (row < N && col < N) {
        data[row * N + col] = 2.0f * curand_uniform(&state) - 1.0f;
    }
}

__global__ void mat_mul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void print_mat(float* m, int N) {
  int size = N * N;
  for (int i = 0; i < size; i++) {
    if (i % N  == 0 && i != 0 ) {
      printf("\n");
    }
    printf("%f, ", m[i]);
  }
}

int main(int argc, char** argv) {
    if (argc != 2) {
      perror("need N x N matrix");
      exit(1);
    }

    size_t N = atoi(argv[1]);
    float *A, *B, *C;

    size_t mat_size = N * N;
    size_t mat_size_bytes = mat_size * sizeof(float);

    CHECK_CUDA(cudaMallocManaged(&A, mat_size_bytes));
    CHECK_CUDA(cudaMallocManaged(&B, mat_size_bytes));
    CHECK_CUDA(cudaMallocManaged(&C, mat_size_bytes));

    dim3 blockSize(32, 32);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
              (N + blockSize.y - 1) / blockSize.y);

    init_mat<<<gridSize, blockSize>>>(A, N, time(NULL));
    CHECK_CUDA(cudaGetLastError());   
    init_mat<<<gridSize, blockSize>>>(B, N, time(NULL));
    CHECK_CUDA(cudaGetLastError());   
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto start = system_clock::now();
    mat_mul<<<gridSize, blockSize>>>(A, B, C, N);
    CHECK_CUDA(cudaGetLastError());   
    CHECK_CUDA(cudaDeviceSynchronize());    
    auto elapsed = duration<double, std::milli>(system_clock::now() - start).count();
    printf("%zux%zu took %lf ms\n", N, N, elapsed);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}