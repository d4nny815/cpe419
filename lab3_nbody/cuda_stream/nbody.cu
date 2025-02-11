#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../common/timer.h"
#include "../common/cuda_help.h"

#define SOFTENING       (1e-9f)

typedef struct { float x, y, z, vx, vy, vz; } Body;

void bodyForceSeq(Body *p, float dt, int n) {
	for (int i = 0; i < n; i++) {
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		for (int j = 0; j < n; j++) {
			float dx = p[j].x - p[i].x;
			float dy = p[j].y - p[i].y;
			float dz = p[j].z - p[i].z;
			float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
			float invDist = 1.0f / sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;

			Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
		}

		p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
	}
}

void integrateSeq(Body *p, float dt, int n) {
  	for (int i = 0; i < n; i++) {
		p[i].x += p[i].vx * dt;
		p[i].y += p[i].vy * dt;
		p[i].z += p[i].vz * dt;
  	}
}

// used ChatGPT for this part of comparing and printing I was lazy
#define EPSILON (1e-3)
void cmp_body_arrs(Body* seq, Body* parallel, size_t n, size_t* bad_calcs, float* max_error) {
    *bad_calcs = 0;
	*max_error = 0;
	for (int i = 0; i < n; i++) {
		float errorx = fabs(parallel[i].x - seq[i].x);
		float errory = fabs(parallel[i].y - seq[i].y);
		float errorz = fabs(parallel[i].z - seq[i].z);
		if (errorx > EPSILON || errory > EPSILON || errorz > EPSILON) {
			*max_error = max(*max_error, max(errorx, max(errory, errorz)));
			(*bad_calcs)++;

			if (*bad_calcs > 3) continue;

			printf("Mismatch at body %d:\n", i);
			printf("  CUDA: (%.6f, %.6f, %.6f)\n", parallel[i].x, parallel[i].y, parallel[i].z);
			printf("  CPU : (%.6f, %.6f, %.6f)\n", seq[i].x, seq[i].y, seq[i].z);
			printf("  Diff: (%.6f, %.6f, %.6f)\n", errorx, errory, errorz);
		}
  	}
}

__global__ void randomizeBodies(Body *data, int n, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, index, 0, &state);

    for (int i = index; i < n; i += stride) {
        data[i].x = 2.0f * curand_uniform(&state) - 1.0f; 
        data[i].y = 2.0f * curand_uniform(&state) - 1.0f;
        data[i].z = 2.0f * curand_uniform(&state) - 1.0f;
        data[i].vx = 2.0f * curand_uniform(&state) - 1.0f;
        data[i].vy = 2.0f * curand_uniform(&state) - 1.0f;
        data[i].vz = 2.0f * curand_uniform(&state) - 1.0f;
    }
}

__global__ void bodyForceKernel(Body *p, float dt, size_t n, size_t start_offset, size_t partition_size) {
    int i = start_offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start_offset + partition_size || i >= n) return;

    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
}

__global__ void integrateKernel(Body *p, float dt, size_t n, size_t start_offset, size_t partition_size) {
    int index = start_offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= start_offset + partition_size || index >= n) return;

    p[index].x += p[index].vx * dt;
    p[index].y += p[index].vy * dt;
    p[index].z += p[index].vz * dt;
}

int main(const int argc, const char** argv) {
    const float dt = 0.01f;
    
    int nIters = 10;
    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);
    if (argc > 2) nIters = atoi(argv[2]);

    int bytes = nBodies * sizeof(Body);

    Body *h_p = (Body*)malloc(bytes);
    Body *d_p;
    CHECK_CUDA(cudaMalloc(&d_p, bytes));

    int deviceId;
	CHECK_CUDA(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
	int sm = prop.multiProcessorCount;
    int num_blocks = 2 * sm;
	int threads_per_block = 4 * MIN_CUDA_THREADS;

    randomizeBodies<<<num_blocks, threads_per_block>>>(d_p, nBodies, time(NULL));
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());

    //Body *p_seq = (Body*)malloc(bytes);
    //CHECK_CUDA(cudaMemcpy(p_seq, d_p, bytes, cudaMemcpyDeviceToHost));

    size_t partition_size = (nBodies + num_blocks - 1) / num_blocks;
    printf("Num Blocks %d, th_p_blk %d, part size %zu\n", 
        num_blocks, threads_per_block, partition_size);

    cudaStream_t* streams = (cudaStream_t*)malloc(num_blocks * sizeof(cudaStream_t));   
    for (int i = 0; i < num_blocks; i++) {
        cudaStreamCreate(&streams[i]);
    }

    double totalTime = 0.0;
    for (int iter = 1; iter <= nIters; iter++) {
        StartTimer();

        for (int i = 0; i < num_blocks; i++) {
            size_t start_offset = i * partition_size;

            bodyForceKernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(d_p, dt, nBodies, start_offset, partition_size);
            CHECK_CUDA(cudaGetLastError());

            integrateKernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(d_p, dt, nBodies, start_offset, partition_size);
            CHECK_CUDA(cudaGetLastError());
    
        }

        for (int i = 0; i < num_blocks; i++) {
            CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        }

        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1) {
            totalTime += tElapsed;
        }
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    }

    CHECK_CUDA(cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost));

    //for (int iter = 1; iter <= nIters; iter++) {
	//	bodyForceSeq(p_seq, dt, nBodies);
	//	integrateSeq(p_seq, dt, nBodies);
  	//}

    //float max_error;
  //  size_t bad_calcs;
  //  cmp_body_arrs(p_seq, h_p, nBodies, &bad_calcs, &max_error);
//	printf("Num of bad_calcs %zu with max %f \n", bad_calcs, max_error);
//
    double avgTime = totalTime / (double)(nIters - 1);
    float rate = (nIters - 1) / totalTime;
    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
            nIters, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    CHECK_CUDA(cudaFree(d_p));
    free(h_p);
    //free(p_seq);

    for (int i = 0; i < num_blocks; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
