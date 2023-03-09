#include <functional>

#define BENCHMARK_COUNT 10u

typedef const std::function<void()> VoidCallback;

void run_cpu_benchmark(VoidCallback& exec);
void run_gpu_benchmark(VoidCallback& setup, VoidCallback& exec);
