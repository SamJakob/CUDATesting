# Set up CMake and project.
cmake_minimum_required(VERSION 3.24)
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_FLAGS "-Ikernels/")

# Project Specification
project(CUDATesting CUDA)
add_executable(CUDATesting src/main.cu src/benchmark.cu src/kernels/histograms.cu src/kernels/array.cu)

set_target_properties(CUDATesting PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
