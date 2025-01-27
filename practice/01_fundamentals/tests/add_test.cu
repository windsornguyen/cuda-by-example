/*--------------------------------------------------------------------*/
/* add_test.cu                                                        */
/* Author: GPT-4o                                                     */
/*--------------------------------------------------------------------*/

#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA_ERROR(call) {                                       \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl;         \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Function signatures from implementation
__global__ void sequentialAddKernel(float *a, float *b, float *c, int n);
__global__ void addKernel(float *a, float *b, float *c, int n);

// RAII class for managing CUDA memory
class CudaMemory {
public:
    CudaMemory(size_t size) { CHECK_CUDA_ERROR(cudaMalloc(&ptr, size)); }
    ~CudaMemory() { cudaFree(ptr); }
    void* get() const { return ptr; }

private:
    void* ptr;
};

// Utility function to run the kernel and validate results
void runKernelAndValidate(float* h_a, float* h_b, float* h_expected, int n, bool useSequential = false) {
    CudaMemory d_a(n * sizeof(float)), d_b(n * sizeof(float)), d_c(n * sizeof(float));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a.get(), h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b.get(), h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    if (useSequential) {
        sequentialAddKernel<<<1, 1>>>(static_cast<float*>(d_a.get()), static_cast<float*>(d_b.get()), static_cast<float*>(d_c.get()), n);
    } else {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(d_a.get()), static_cast<float*>(d_b.get()), static_cast<float*>(d_c.get()), n);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float* h_c = new float[n];
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c.get(), n * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        assert(h_c[i] == h_expected[i]);
    }
    delete[] h_c;
}

// Test cases
void testAddKernelBasic() {
    int n = 5;
    float h_a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float h_expected[] = {6.0, 6.0, 6.0, 6.0, 6.0};

    runKernelAndValidate(h_a, h_b, h_expected, n);
    runKernelAndValidate(h_a, h_b, h_expected, n, true);  // Test sequential kernel
    printf("Basic functionality test passed!\n");
}

void testAddKernelEdgeCases() {
    // Empty vector case
    int n = 0;
    float h_a[] = {};
    float h_b[] = {};
    float h_expected[] = {};

    runKernelAndValidate(h_a, h_b, h_expected, n);
    runKernelAndValidate(h_a, h_b, h_expected, n, true);
    printf("Edge case: Empty vector test passed!\n");

    // Single element case
    n = 1;
    float h_a_single[] = {3.0f};
    float h_b_single[] = {4.0f};
    float h_expected_single[] = {7.0f};

    runKernelAndValidate(h_a_single, h_b_single, h_expected_single, n);
    runKernelAndValidate(h_a_single, h_b_single, h_expected_single, n, true);
    printf("Edge case: Single element test passed!\n");

    // Odd number of elements
    n = 7;
    float h_a_odd[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float h_b_odd[] = {7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float h_expected_odd[] = {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f};

    runKernelAndValidate(h_a_odd, h_b_odd, h_expected_odd, n);
    runKernelAndValidate(h_a_odd, h_b_odd, h_expected_odd, n, true);
    printf("Edge case: Odd number of elements test passed!\n");

    // Large and small values
    n = 3;
    float h_a_large[] = {1e30f, -1e30f, 1e-30f};
    float h_b_large[] = {1e30f, -1e30f, 1e-30f};
    float h_expected_large[] = {2e30f, -2e30f, 2e-30f};

    runKernelAndValidate(h_a_large, h_b_large, h_expected_large, n);
    runKernelAndValidate(h_a_large, h_b_large, h_expected_large, n, true);
    printf("Edge case: Large and small values test passed!\n");
}

void testAddKernelScalability() {
    int n = 1 << 20; // 1 million elements
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_expected = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(n - i);
        h_expected[i] = static_cast<float>(n);
    }

    runKernelAndValidate(h_a, h_b, h_expected, n);
    // Note: We don't test the sequential kernel here due to performance reasons

    delete[] h_a;
    delete[] h_b;
    delete[] h_expected;

    printf("Scalability test passed!\n");
}

void testNonPowerOfTwo() {
    int n = 1007; // Non-power-of-two length
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_expected = new float[n];

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
        h_expected[i] = h_a[i] + h_b[i];
    }

    runKernelAndValidate(h_a, h_b, h_expected, n);
    runKernelAndValidate(h_a, h_b, h_expected, n, true);

    delete[] h_a;
    delete[] h_b;
    delete[] h_expected;

    printf("Non-power-of-two test passed!\n");
}

// Function to run all tests
void runAllTests() {
    testAddKernelBasic();
    testAddKernelEdgeCases();
    testAddKernelScalability();
    testNonPowerOfTwo();
    printf("All tests passed!\n");
}