#include <sycl/sycl.hpp>
#include <array>
#include <cstdlib>
#include <iostream>
#include <cmath>

using namespace sycl;
using namespace std;

void init_arrs(queue& q, float* arr, const size_t width, const size_t height) {
    std::time_t epoch_time = std::time(nullptr);

    q.submit([&](handler& h) {
        h.parallel_for(range<1>(width * height), [=](auto i){
            uint32_t seed = static_cast<uint32_t>(epoch_time * 6242878123) + i[0]; 
            seed = (1664525 * seed + 1013904223) % 0xFFFFFFFF;

            // Convert to float in range [0, 100]
            arr[i] = static_cast<float>(seed) / static_cast<float>(0xFFFFFFFF) * 100.0f;
        });
    }).wait();
}

void matmul_seq(float const* arr1, float const* arr2, float* res_arr, 
        const size_t width, const size_t height) {
	
    for (size_t row = 0; row < height; row++){
        for(size_t col = 0; col < height; col++) {
            size_t ind = row * height + col;
            res_arr[ind] = 0.0f;
            for(size_t k = 0; k < width; k++){
                res_arr[ind] += arr1[row * width + k] * arr2[k * height + col];
            }
        }
	}
}

void matmul_par(queue& q, float const* arr1, float const* arr2, float* res_arr, 
        const size_t width, const size_t height) {
	
    q.submit([&](handler& h) {
        h.parallel_for(range<2>(height, height), [=](id<2> i){
            auto row = i[0];
            auto col = i[1];
            
            float sum = 0.0f;
            for(size_t k = 0; k < width; k++){
                sum += arr1[row * width + k] * arr2[k * height + col];
            }
            res_arr[row * height + col] = sum;
        });
    }).wait();
}

bool equal_arrs(float const* arr1, float const* arr2, 
        const size_t width, const size_t height) {
    
    constexpr float EPISLON = 1e-2; 
    for (size_t row = 0; row < height; row++){
        for(size_t col = 0; col < width; col++) {
            if (std::abs(arr1[row * width + col] - arr2[row * width + col]) > EPISLON) {
                fprintf(stderr, "[%zu] %f != %f\n", row * width + col, 
                    arr1[row * width + col], arr2[row * width + col]);
                return false;
            }
        }
	}
    return true;
}

void print_arr(float* arr1, size_t width, size_t height) {
    for (size_t row = 0; row < height; row++){
        for(size_t col = 0; col < width; col++) {
            cout << arr1[row * width + col] << ' ';
        }
        cout << std::endl;
	}
}


int main(void) {
    // queue q(default_selector_v, exception_handler);
    constexpr size_t M = 10;
    constexpr size_t N = 5;

    auto selector = default_selector_v;
    // auto selector = gpu_selector_v;

    queue q(selector);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";


    float* mat1 = malloc_shared<float>(M * N, q);
    float* mat2 = malloc_shared<float>(M * N, q);
    if ((mat1 == nullptr) || (mat2 == nullptr)) {
        if (mat1 != nullptr) free(mat1, q);
        if (mat2 != nullptr) free(mat2, q);

        cout << "Shared memory allocation failure.\n";
        return -1;
    }

    init_arrs(q, mat1, M, N);
    init_arrs(q, mat2, N, M);

    // print_arr(mat1, M, N);
    // printf("\n===========\n\n");
    // print_arr(mat2, N, M);
    // printf("\n===========\n\n");

    float* seq = (float*)malloc(N * N * sizeof(float));
    float* par = malloc_shared<float>(N * N, q);

    if ((seq == nullptr) || (par == nullptr)) {
        if (seq != nullptr) free(seq, q);
        if (par != nullptr) free(par, q);

        cout << "Shared memory allocation failure.\n";
        return -1;
    }

    matmul_seq(mat1, mat2, seq, M, N);
    matmul_par(q, mat1, mat2, par, M, N);

    // print_arr(seq, N, N);
    // printf("\n===========\n\n");
    // print_arr(par, N, N);

    equal_arrs(seq, par, N, N);

    free(mat1, q);
    free(mat2, q);
    free(seq);
    free(par, q);

    cout << "Successfully completed on device.\n";
    return 0;
}