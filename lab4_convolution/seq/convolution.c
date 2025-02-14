#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"

#define MAX_VAL         (255)
#define KERNEL_SIZE     (3)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define WRAP_INDEX(idx, max) (((idx) + (max)) % (max))


void init_arr(int* data, size_t n);
void print_arr(int* data, size_t Nx, size_t Ny);

void get_window(int* arr, int x, int y, size_t Nx, size_t Ny, 
    int* window, size_t kernel_size);
int convolve(int* window, float* kernel, size_t kernel_size);

int main(const int argc, const char** argv) {
    size_t Nx = 100;
    size_t Ny = 100;
    if (argc == 2) {
        Nx = atoi(argv[1]);
    }
    else if (argc == 3) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
    }

    size_t mat_size = Nx * Ny;
    size_t bytes = mat_size * sizeof(int);
    
    int* frame = (int*)malloc(bytes);
    if (!frame) {
        perror("malloc failed\n");
        exit(1);
    }
    int* end_frame = (int*)malloc(bytes);
    if (!frame) {
        perror("malloc failed\n");
        exit(1);
    }

    init_arr(frame, mat_size);
    
    float KERNEL[KERNEL_SIZE * KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i ++) KERNEL[i] = 0.5f;

    StartTimer();

    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            int window[KERNEL_SIZE * KERNEL_SIZE];
            get_window(frame, j, i, Nx, Ny, window, KERNEL_SIZE);
            end_frame[j * Nx + i] = convolve(window, KERNEL, KERNEL_SIZE);
        }
    }

    const double tElapsed = GetTimer() / 1000.0;

    printf("%zu, %zu, %lf\n", Nx, Ny, tElapsed);

    #ifdef DEBUG
    print_arr(frame, Nx, Ny);
    printf("\n==============\n\n");
    print_arr(end_frame, Nx, Ny);
    #endif

    free(frame);
    free(end_frame);

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
            printf("%d, ", data[j * Nx + i]);
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
