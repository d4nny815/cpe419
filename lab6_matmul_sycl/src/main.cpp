#include <sycl/sycl.hpp>
#include <array>
#include <cstdlib>
#include <iostream>

using namespace sycl;
using namespace std;

void init_arrs(queue& q, float* arr, size_t width, size_t height) {
    range num_items{width * height};

    auto e = q.parallel_for(range(width, height), [=](auto index) { 
        size_t row = index[0];
        size_t col = index[1];
        arr[row * width + col] = 1.0f; 
    });

    // for (size_t row = 0; row < height; row++){
    //     for(size_t col = 0; col < width; col++) {
    //         arr[row * width + col] = std::rand();
    //     }
	// }
    e.wait();
}

void matmul_seq(float* arr1, float* arr2, float* res_arr, size_t width, size_t height) {
	for (size_t row = 0; row < height; row++){
        for(size_t col = 0; col < width; col++) {
            float sum = 0;
            for(size_t k = 0; k < width; k++){
                sum += arr1[row * width + k] * arr2[k * width + col];
            }
            res_arr[row * width + col] = sum;
        }
	}
}

bool equal_arrs(float** arr1, float** arr2, size_t width, size_t height) {
    for (size_t row = 0; row < height; row++){
        for(size_t col = 0; col < width; col++) {
            if (arr1[row * width + col] != arr2[row * width + col]) {
                cerr << "Arrays not equal.\n";
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
    constexpr size_t ARR_WIDTH = 10;
    constexpr size_t ARR_HEIGHT = 4;

    queue q(default_selector_v);

    std::srand(static_cast<unsigned>(std::time(nullptr)));


    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";


    float* sequential = malloc_shared<float>(ARR_WIDTH * ARR_HEIGHT, q);
    float* parallel = malloc_shared<float>(ARR_WIDTH * ARR_HEIGHT, q);

    if ((sequential == nullptr) || (parallel == nullptr)) {
        if (sequential != nullptr) free(sequential, q);
        if (parallel != nullptr) free(parallel, q);

        cout << "Shared memory allocation failure.\n";
        return -1;
    }

    init_arrs(q, sequential, ARR_WIDTH, ARR_HEIGHT);
    print_arr(sequential, ARR_WIDTH, ARR_HEIGHT);


    // seq_kernel(sequential, ARRAY_SIZE);

    // parallel_kernel(q, parallel, ARRAY_SIZE);

    // if (!equal_arrs(sequential, parallel, ARRAY_SIZE)) {
    //     exit(1);
    // }    

    printf("made it here\n");
    free(sequential, q);
    free(parallel, q);

    cout << "Successfully completed on device.\n";
    return 0;
}