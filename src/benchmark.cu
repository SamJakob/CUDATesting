#include <iostream>
#include <chrono>
#include <numeric>

#include "benchmark.h"

void run_cpu_benchmark(VoidCallback& exec) {
    std::chrono::time_point<std::chrono::steady_clock> t1, t2;
    double benchmarks[BENCHMARK_COUNT] = {};

    for (double& benchmark : benchmarks) {
        t1 = std::chrono::high_resolution_clock::now();

        exec();

        t2 = std::chrono::high_resolution_clock::now();
        benchmark = (std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1)).count();
    }

    auto averageInvocationTime = std::accumulate(benchmarks, std::end(benchmarks), 0.0) / std::size(benchmarks);
    std::cout << "Average Invocation Time: " << averageInvocationTime << "ms" << std::endl;
    for (auto benchmark : benchmarks) std::cout << benchmark << std::endl;
}

void run_gpu_benchmark(VoidCallback& setup, VoidCallback& exec) {
    cudaEvent_t kernelStart, kernelStop;

    float benchmarks[BENCHMARK_COUNT] = {};

    for (float& benchmark : benchmarks) {
        setup();
        cudaDeviceSynchronize();

        cudaEventCreate(&kernelStart);
        cudaEventCreate(&kernelStop);
        cudaEventRecord(kernelStart, nullptr);

        exec();

        cudaEventRecord(kernelStart, nullptr);
        cudaEventElapsedTime(&benchmark, kernelStart, kernelStop);

        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
    }

    auto averageInvocationTime = std::accumulate(benchmarks, std::end(benchmarks), 0.0) / std::size(benchmarks);
    std::cout << "Average Invocation Time: " << averageInvocationTime << "ms" << std::endl;
    for (auto benchmark : benchmarks) std::cout << benchmark << std::endl;
}
