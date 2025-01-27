/*--------------------------------------------------------------------*/
/* 01_kernel_call.cu                                                  */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

#include <iostream>

__global__ void kernel() {
    // Runs on GPU, launched on CPU
}

int main() {
    kernel<<<1, 1>>>(); // Launch the kernel with 1 block, 1 thread
    printf("Hello, world!\n");
    return 0;
}
