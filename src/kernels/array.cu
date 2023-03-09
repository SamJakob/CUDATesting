#include <chrono>
#include <iostream>

#include "kernels.h"
#include "../benchmark.h"

__global__ void run(unsigned int* deviceArrayData) {
    deviceArrayData[threadIdx.x] = threadIdx.x;
}

int boot_array() {
    unsigned int arraySize = 30;
    unsigned int* myArray;

    // Warm-up call (also measure CUDA boot time)
    std::chrono::time_point<std::chrono::steady_clock> t1, t2;
    double benchmark;

    t1 = std::chrono::high_resolution_clock::now();
    cudaMallocManaged(reinterpret_cast<void **>(&myArray), arraySize * sizeof(int));
    cudaMemset(myArray, 0, arraySize * sizeof(int));
    run<<<1, 30>>>(myArray);
    cudaMemcpy(myArray, myArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(&myArray);
    t2 = std::chrono::high_resolution_clock::now();
    benchmark = (std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1)).count();
    std::cout << "CUDA Initialization Time: " << benchmark << "ms" << std::endl;

    // Now benchmark CUDA invocation
    run_cpu_benchmark([&myArray, &arraySize]() {
        cudaMallocManaged(reinterpret_cast<void **>(&myArray), arraySize * sizeof(int));
        cudaMemset(myArray, 0, arraySize * sizeof(int));
        run<<<1, 30>>>(myArray);
        cudaMemcpy(myArray, myArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(&myArray);
    });


    // Print the array afterwards.
    for (unsigned int i = 0; i < arraySize; i++) {
        std::cout << myArray[i] << std::endl;
    }

    return 0;
}