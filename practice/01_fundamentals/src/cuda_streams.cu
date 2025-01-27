/*--------------------------------------------------------------------*/
/* cuda_streams.cu                                                    */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

// Use streams to overlap data transfers and kernel execution

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

 __global__ void addKernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
 }
 

 int main() {
    // Declare streams here
}