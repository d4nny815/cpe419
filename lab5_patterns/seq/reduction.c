#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"
#include <string.h>

#define MAX_VAL         (255)
// #define DEBUG

void init_arr(double* data, size_t n);
void print_arr(double* data, size_t N);
double reduction_neighbor(double* data, size_t N);
double reduction_stride(double* data, size_t N);

int main(const int argc, const char** argv) {
    size_t N = 10;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    size_t bytes = N * sizeof(double);
    double* arr_neighbors = (double*)malloc(bytes);
    double* arr_stride = (double*)malloc(bytes);
    if (!arr_stride || !arr_neighbors) {
        perror("malloc failed");
        exit(1);
    }

    init_arr(arr_neighbors, N);
    memcpy(arr_stride, arr_neighbors, bytes);

    #ifdef DEBUG
    print_arr(arr_neighbors, N);
    print_arr(arr_stride, N);
    #endif

    StartTimer();

    double neighbor = reduction_neighbor(arr_neighbors, N);

    const double t_neighbor = GetTimer() / 1000.0;

    
    StartTimer();

    double stride = reduction_stride(arr_stride, N);

    const double t_stride = GetTimer() / 1000.0;

    printf("%zu, %lf, %lf, %lf, %lf\n", N, neighbor, t_neighbor, stride, t_stride);

    free(arr_neighbors);
    free(arr_stride);

    return 0;
}

double reduction_neighbor(double* data, size_t N) {
    while (N > 1) {
        size_t i, newN = N / 2;
    
        for (i = 0; i < N - 1; i += 2) {
            data[i / 2] = data[i] + data[i + 1];
        }
    
        if (i < N) {
            data[i / 2] = data[i] + (i + 1 < N ? data[i + 1] : 0);
            newN++;
        }
    
        #ifdef DEBUG
        printf("N = %zu ===> ", newN);
        print_arr(data, newN);  
        #endif
    
        N = newN; 
    }
    return data[0];
}

double reduction_stride(double* data, size_t N) {
    while (N > 1) {
        size_t stride = (N + 1) / 2; 
       
        #pragma omp parallel for schedule(static)  
        for (size_t i = 0; i < stride; i++) {
            if (i + stride < N) {
                data[i] += data[i + stride];
            }
        }

        #ifdef DEBUG
        printf("N = %zu ===> ", stride);
        print_arr(data, stride);
        #endif

        N = stride; 
    }

    return data[0]; 
}

void init_arr(double* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = rand();
        // data[i] = i;
    }
}

void print_arr(double* data, size_t N) {
    for (int i = 0; i < N; i++) {
        printf("%lf, ", data[i]);
    }
    printf("\n");
}


