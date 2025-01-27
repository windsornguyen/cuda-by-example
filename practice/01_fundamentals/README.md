# Fundamentals of CUDA C

In this module, you'll learn the fundamentals of programming in CUDA C.

1. add.cu:
- CUDA kernels are executed by many threads in parallel

- Each thread should work on a subset of the data

- Learn how to use threadIdx, blockIdx, and blockDim to calculate a unique index for each thread
    - Example: int idx = blockIdx.x * blockDim.x + threadDim.x;

- If there are more elements than threads, use a loop and stride by grids
    - Example: for (int i = idx; i < n; i += blockDim.x * gridDim.x) { ... }

- Make sure threads don't access memory outside the array

- Decide on appropriate block size, e.g. 256 threads per block

- Calculate grid size based on number of elements and block size

- You can allocate memory on the device (GPU) using `cudaMalloc`

- You can transfer data between host and device using `cudaMemcpy`

- You should check for errors after the kernel launches and after memory operations
    - Example: Use `cudaGetLastError()` and `cudaDeviceSynchronize()`

