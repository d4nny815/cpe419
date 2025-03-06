#include <sycl/sycl.hpp>
#include <array>
#include <iostream>

using namespace sycl;
using namespace std;

void seq_kernel(float* arr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        arr[i] = i * 2.0;
    }
}

void parallel_kernel(queue& q, float* a, size_t size) {
    range num_items{size};
  
    auto e = q.parallel_for(num_items, [=](auto i) { 
        a[i] = i * 2.0; 
    });
  
    e.wait();
}

bool equal_arrs(float* arr1, float* arr2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            cout << "Failed on device.\n";
            return false;
        }
    }
    return true;
}



int main(void) {
    // queue q(default_selector_v, exception_handler);
    constexpr size_t ARRAY_SIZE = 1000000;

    queue q(default_selector_v);

    // Print out the device information used for the kernel code.
    cout << "Running on device: "
        << q.get_device().get_info<info::device::name>() << "\n";
    cout << "Array size: " << ARRAY_SIZE << "\n";

    float* sequential = malloc_shared<float>(ARRAY_SIZE, q);
    float* parallel = malloc_shared<float>(ARRAY_SIZE, q);

    if ((sequential == nullptr) || (parallel == nullptr)) {
        if (sequential != nullptr) free(sequential, q);
        if (parallel != nullptr) free(parallel, q);

        cout << "Shared memory allocation failure.\n";
        return -1;
    }

    seq_kernel(sequential, ARRAY_SIZE);

    parallel_kernel(q, parallel, ARRAY_SIZE);

    if (!equal_arrs(sequential, parallel, ARRAY_SIZE)) {
        exit(1);
    }    

    free(sequential, q);
    free(parallel, q);

    cout << "Successfully completed on device.\n";
    return 0;
}