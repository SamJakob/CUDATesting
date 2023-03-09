#include <algorithm>
#include <random>
#include <iostream>

#include "../benchmark.h"

// If we can locate kernels.h include it, otherwise skip this step.
// IF THIS IS CAUSING ERRORS, PLEASE SIMPLY DELETE THESE LINES.
#if __has_include("kernels.h")
#include "kernels.h"
#endif

// The number of bins to define for the histograms.
#define BIN_COUNT 4

// The number of entries in the input array. Here is
// 2**32.
//#define INPUT_COUNT 1 << 32
#define INPUT_COUNT 4

/**
 * A kernel to compute a histogram based on a set of bins and an input array.
 * This computes the values and stores them in device memory.
 *
 * @param device_bins
 * @param device_input
 */
//__global__ void histogram_kernel_naive_old(unsigned int* device_bins,
//                                       const unsigned int* device_input) {
//
//    // Calculate our position, and the item we're dealing with.
//    auto position = blockDim.x * blockIdx.x + threadIdx.x;
//    if (position >= INPUT_COUNT) return;
//
//    auto item = device_input[position];
//
//    // Figure out which 'bin' our item belongs in, then increment that bin.
//    auto bin = item % BIN_COUNT;
//    device_bins[bin]++;
//
//}

__global__ void histogram_kernel_naive(unsigned int* device_bins,
                                       const unsigned int* device_input) {

    // Calculate our position, and the item we're dealing with.
    auto position = blockDim.x * blockIdx.x + threadIdx.x;

    // Total number of threads.
    auto stride = blockDim.x * gridDim.x;

    while (position < BIN_COUNT) {
        atomicAdd(&device_bins[position], 1);
        position += stride;
    }

}

/**
 * Allocates an array of unsigned integers, of size N, and generates
 * uniformly random values for the array. The pointer to the newly allocated
 * array is returned.
 *
 * @param N The number of elements to generate.
 * @return The pointer to the allocated and generated array.
 */
static void generate_data(unsigned int** destination, size_t N) {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_int_distribution<> distribution(0, BIN_COUNT);

    // Allocate and generate random values for an array up to size N.
    // Computed using C++ 11's random library with a uniform integer distribution from 0 to BIN_COUNT.
    if (*destination == nullptr) {
        *destination = (unsigned int*) malloc(sizeof(unsigned int) * N);
    }

    for (unsigned int i = 0; i < N; i++) {
//        (*destination)[i] = distribution(generator);
        (*destination)[i] = 1;
    }

    std::cout << "Preview of generated data:" << std::endl;
    for (size_t i = std::min(N - 10, (size_t) 0); i < N; i++) std::cout << (*destination)[i] << (i < N - 1 ? ", " : "");
    std::cout << std::endl << std::endl;
}

int boot_histograms() {
    std::cout
        << "Benchmarking histograms kernel for " << INPUT_COUNT << " array value(s) (" << BIN_COUNT << " bucket(s))."
        << std::endl
        << std::endl;

    unsigned int
        * host_input = nullptr,
        * host_bins = nullptr,
        * device_input = nullptr,
        * device_bins = nullptr;

    std::cout << "Synchronized GPU. Launching benchmark." << std::endl;

    run_gpu_benchmark(
        [&device_input, &device_bins, &host_input, &host_bins]() mutable {
            // Allocate and generate host data.
            generate_data(&host_input, INPUT_COUNT);

            if (host_bins == nullptr) host_bins = (unsigned int*) calloc(BIN_COUNT, sizeof(unsigned int));

            // Allocate GPU memory.
            cudaMalloc((void**) &device_input, sizeof(unsigned int) * INPUT_COUNT);
            cudaMalloc((void**) &device_bins, sizeof(unsigned int) * BIN_COUNT);

            // Copying data from host to GPU buffers.
            cudaMemcpy(device_input, host_input, sizeof(unsigned int) * INPUT_COUNT, cudaMemcpyHostToDevice);
            cudaMemcpy(device_bins, host_bins, sizeof(unsigned int) * BIN_COUNT, cudaMemcpyHostToDevice);
        },
        [&device_input, &device_bins, &host_bins]() mutable {
            histogram_kernel_naive<<<8, 512, sizeof(unsigned int) * BIN_COUNT>>>(device_bins, device_input);
            cudaMemcpy(host_bins, device_bins, sizeof(unsigned int) * BIN_COUNT, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
    );

    for (int i = 0; i < BIN_COUNT; i++) {
        printf("%d,", host_bins[i]);
    }
    printf("\n");

    return 0;
}

// If this file is standalone (i.e., if IS_COMPONENT is not set), automatically include a
// main method.
#ifndef IS_COMPONENT
int main(const int argc, const char** argv) {
    return boot_histograms();
}
#endif
