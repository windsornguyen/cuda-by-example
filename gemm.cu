#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

// Error-checking macro for CUDA calls
#define CUDA_CHECK(err) if(err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); }

// Error-checking macro for cuBLAS calls
#define CUBLAS_CHECK(err) if(err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error: " << err << std::endl; exit(EXIT_FAILURE); }

// Naive CUDA kernel for GEMM
__global__ void sgemm_naive(
    int num_rows_A, 
    int num_cols_B, 
    int shared_dim, 
    float alpha, 
    const float *A, 
    const float *B, 
    float beta,
    float *C) 
{
    // So we have A @ B => (u, v) @ (w, x) = (u', x')
    // row_C corresponds to u, col_C corresponds to x
    const uint row_C = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col_C = blockIdx.y * blockDim.y + threadIdx.y;

    // We want to keep going until either
    // 1. row_C is past num_rows_A (u' <= u)
    // 2. col_C is past num_cols_B (x' <= x)
    if (row_C < num_rows_A && col_C < num_cols_B) {
        // Safe zone! Proceed with the matmul per usual
        float val = 0.0;
        for (int i = 0; i < shared_dim; ++i) {
            val += A[row_C * shared_dim + i] * B[i * num_cols_B + col_C];
        }
        C[row_C * num_cols_B + col_C] = alpha * val + beta * C[row_C * num_cols_B + col_C];
    }
}

// Function to initialize maxatrices with random values
void initialize_matrix(float* matrix, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Matrix dimensions
    int M = 1024; // Rows in A and C
    int N = 1024; // Columns in B and C
    int K = 1024; // Columns in A, rows in B

    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_naive = (float*)malloc(M * N * sizeof(float));
    float *h_C_cublas = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    initialize_matrix(h_A, M * K);
    initialize_matrix(h_B, K * N);
    initialize_matrix(h_C_naive, M * N);
    initialize_matrix(h_C_cublas, M * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_naive, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, M * N * sizeof(float)));

    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_naive, h_C_naive, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C_cublas, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Define CUDA grid and block dimensions
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, 
                    (M + threads_per_block.y - 1) / threads_per_block.y);

    // Benchmark naive GEMM kernel
    auto start_naive = std::chrono::high_resolution_clock::now();
    sgemm_naive<<<num_blocks, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C_naive);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> naive_duration = end_naive - start_naive;
    std::cout << "Naive GEMM execution time: " << naive_duration.count() << " ms" << std::endl;

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Benchmark cuBLAS GEMM
    auto start_cublas = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_cublas = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cublas_duration = end_cublas - start_cublas;
    std::cout << "cuBLAS GEMM execution time: " << cublas_duration.count() << " ms" << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C_naive, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results for correctness
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = fmax(max_error, fabs(h_C_naive[i] - h_C_cublas[i]));
    }
    std::cout << "Max error between naive and cuBLAS: " << max_error << std::endl;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_cublas));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_cublas);

    return 0;
}
