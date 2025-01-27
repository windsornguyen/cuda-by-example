/*--------------------------------------------------------------------*/
/* 02_passing_params.cu                                               */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

#include <iostream>
#include "../book.h"

// You can define kernels as you'd normally do for regular functions in C 
__global__ void add(int a, int b, int *c) {
    *c = a + b;
}

int main() {
    int c;
    int *dev_c;

    // Alloc memory on the DEVICE, basically identical to malloc
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
    // Here, we pass in ptr to addr of newly allocated memory and the size of the allocation

    add<<<1, 1>>>(2, 7, dev_c);
    HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

    printf("2 + 7 = %d\n", c);

    cudaFree(dev_c); // Have to use cudaFree if using cudaMalloc
    return 0;
}
