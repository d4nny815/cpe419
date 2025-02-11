#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../common/timer.h"
#include "../common/cuda_help.h"

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;


__global__ void randomizeBodies(Body *data, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
    for (int i = index; i < n; i += stride) {
        // printf("hi from thread: %d\n", i);
        data[i].x = 2.0f * i * i;
        data[i].y = 2.0f * i * i;
        data[i].z = 2.0f * i * i;
        data[i].vx = 0.0f;
        data[i].vy = 0.0f;
        data[i].vz = 0.0f;
    }
}

__global__ void bodyForceKernel(Body *p, float dt, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
    for (int i = index; i < n; i += stride) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

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
}

__global__ void integrateKernel(Body *p, float dt, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
    for (int i = index; i < n; i += stride) {
        // printf("i = %d\n", i);
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

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

int main(const int argc, const char** argv) {
    const float dt = 0.01f;
    
    int nIters = 10;
    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);
    if (argc > 2) nIters = atoi(argv[2]);

    int bytes = nBodies * sizeof(Body);

    Body *p = (Body*)malloc(bytes);
    CHECK_CUDA(cudaMallocManaged(&p, bytes));

    int deviceId;
	CHECK_CUDA(cudaGetDevice(&deviceId));
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
	int sm = prop.multiProcessorCount;
    int num_blocks = 2 * sm;
	int threads_per_block = 4 * MIN_CUDA_THREADS;

    randomizeBodies<<<num_blocks, threads_per_block>>>(p, nBodies);
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());

    //Body *p_seq = (Body*)malloc(bytes);
    //CHECK_CUDA(cudaMemcpy(p_seq, p, bytes, cudaMemcpyDeviceToHost));
    
    double totalTime = 0.0;

    for (int iter = 1; iter <= nIters; iter++) {
        StartTimer();

        bodyForceKernel<<<num_blocks, threads_per_block>>>(p, dt, nBodies);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        integrateKernel<<<num_blocks, threads_per_block>>>(p, dt, nBodies);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        const double tElapsed = GetTimer() / 1000.0;
        if (iter > 1) {
            totalTime += tElapsed;
        }
        printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
    }

    //for (int iter = 1; iter <= nIters; iter++) {
	//	bodyForceSeq(p_seq, dt, nBodies);
	//	integrateSeq(p_seq, dt, nBodies);
  	//}

    // for (int i = 0 ; i < nBodies; i++) { // integrate position
    //     printf("i: %d vx: %f vy: %f vz: %f start %d  \n",
    //         i, h_p[i].x, h_p[i].y, h_p[i].z, 0);
    // }

    //float max_error;
    //size_t bad_calcs;
    //cmp_body_arrs(p_seq, p, nBodies, &bad_calcs, &max_error);
	//printf("Num of bad_calcs %zu with max %f \n", bad_calcs, max_error);

    double avgTime = totalTime / (double)(nIters - 1);
    float rate = (nIters - 1) / totalTime;
    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
            nIters, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    CHECK_CUDA(cudaFree(p));
    //free(p_seq);

    return 0;
}
