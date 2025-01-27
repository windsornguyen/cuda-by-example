/*--------------------------------------------------------------------*/
/* add.cu                                                             */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

/**
 * @brief CUDA kernel to add two vectors element-wise, sequentially.
 *
 * This kernel adds two vectors `a` and `b` of length `n`, storing the result
 * in vector `c`. Each thread computes one element of the result.
 *
 * @param a Pointer to the first input vector of floats.
 * @param b Pointer to the second input vector of floats.
 * @param c Pointer to the output vector of floats where the result will be stored.
 * @param n The number of elements in the vectors.
 */
__global__ void sequentialAddKernel(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i]; // Very slow!
    }
}


/**
 * @brief CUDA kernel to add two vectors element-wise in parallel.
 *
 * This kernel adds two vectors `a` and `b` of length `n`, storing the result
 * in vector `c`. Each thread computes one element of the result.
 *
 * @param a Pointer to the first input vector of floats.
 * @param b Pointer to the second input vector of floats.
 * @param c Pointer to the output vector of floats where the result will be stored.
 * @param n The number of elements in the vectors.
 */
__global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// This might help you visualize blockIdx, blockDim, and threadIdx:
// +-----------------+-----------------+-----------------+
// |  Block (0,1)     |  Block (1,1)     |  Block (2,1)    |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// |  | T0| T1|       |  | T0| T1|       |  | T0| T1|      |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// |  | T2| T3|       |  | T2| T3|       |  | T2| T3|      |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// +-----------------+-----------------+-----------------+

// +-----------------+-----------------+-----------------+
// |  Block (0,0)     |  Block (1,0)     |  Block (2,0)    |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// |  | T0| T1|       |  | T0| T1|       |  | T0| T1|      |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// |  | T2| T3|       |  | T2| T3|       |  | T2| T3|      |
// |  +---+---+       |  +---+---+       |  +---+---+      |
// +-----------------+-----------------+-----------------+

