#ifndef CUDA_HELP
#define CUDA_HELP

#define MIN_CUDA_THREADS (32)

#define CHECK_CUDA(call) \
	if ((call) != cudaSuccess) { \
		  fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); \
		  exit(EXIT_FAILURE); \
	}

#endif /* cuda_help.h */
