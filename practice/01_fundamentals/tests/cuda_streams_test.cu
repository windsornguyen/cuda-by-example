/*--------------------------------------------------------------------*/
/* cuda_streams_test.cu                                               */
/* Author: GPT-4o                                                     */
/*--------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_CUDA_ERROR(call) {                                       \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl;         \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Kernel that performs a simple operation (multiply by 2)
__global__ void simpleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Artificial work to make the kernel take longer
        for (int i = 0; i < 250; ++i) {
            data[idx] = data[idx] * 2.0f;
        }
    }
}

// Function to run kernels without streams
float runWithoutStreams(float* d_data, int n, int numKernels) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numKernels; ++i) {
        simpleKernel<<<(n + 255) / 256, 256>>>(d_data + i * n, n);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count();
}

// Function to run kernels with streams
float runWithStreams(float* d_data, int n, int numKernels, int numStreams) {
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numKernels; ++i) {
        simpleKernel<<<(n + 255) / 256, 256, 0, streams[i % numStreams]>>>(d_data + i * n, n);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }

    return duration.count();
}

int main() {
    const int n = 1 << 20;  // 1M elements per kernel
    const int numKernels = 8;  // Number of kernel launches
    const int numStreams = 4;  // Number of streams to use

    // Allocate device memory
    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * numKernels * sizeof(float)));

    // Initialize data
    std::vector<float> h_data(n * numKernels, 1.0f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), n * numKernels * sizeof(float), cudaMemcpyHostToDevice));

    // Run without streams
    float timeWithoutStreams = runWithoutStreams(d_data, n, numKernels);
    std::cout << "Time without streams: " << timeWithoutStreams << " ms" << std::endl;

    // Run with streams
    float timeWithStreams = runWithStreams(d_data, n, numKernels, numStreams);
    std::cout << "Time with " << numStreams << " streams: " << timeWithStreams << " ms" << std::endl;

    // Check if streams version is faster
    if (timeWithStreams >= timeWithoutStreams) {
        std::cerr << "Error: Streams version is not faster!" << std::endl;
        std::cerr << "Without streams: " << timeWithoutStreams << " ms" << std::endl;
        std::cerr << "With streams: " << timeWithStreams << " ms" << std::endl;
        return 1;
    }

    float speedup = timeWithoutStreams / timeWithStreams;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_data));

    std::cout << "Test passed successfully!" << std::endl;
    return 0;
}
