/*--------------------------------------------------------------------*/
/* 03_querying_devices.cu                                             */
/* Author: Windsor Nguyen                                             */
/*--------------------------------------------------------------------*/

#include "../book.h"

int main() {
    cudaDeviceProp prop;
    int count;

    // Get the number of CUDA-capable devices
    HANDLE_ERROR(cudaGetDeviceCount(&count));

    // Loop through all devices and query properties
    for (int i = 0; i < count; ++i) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        
        // Print general information
        printf(" ------------------------------------------------\n");
        printf("       General Information for device %d\n", i);
        printf(" ------------------------------------------------\n");
        printf(" Name:                                %s\n", prop.name);
        printf(" Compute capability:                  %d.%d\n", prop.major, prop.minor); 
        printf(" Clock rate:                          %d kHz\n", prop.clockRate);
        printf(" Device copy overlap:                 %s\n", prop.deviceOverlap ? "Enabled" : "Disabled");
        printf(" Kernel execution timeout:            %s\n", prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

        // Print memory information
        printf(" ------------------------------------------------\n");
        printf("       Memory Information for device %d\n", i);
        printf(" ------------------------------------------------\n");
        printf(" Total global memory:                 %zu bytes\n", prop.totalGlobalMem);
        printf(" Total constant memory:               %zu bytes\n", prop.totalConstMem);
        printf(" Max memory pitch:                    %zu bytes\n", prop.memPitch);
        printf(" Texture alignment:                   %zu bytes\n", prop.textureAlignment);

        // Print multiprocessor information
        printf(" ------------------------------------------------\n");
        printf("       MP Information for device %d\n", i);
        printf(" ------------------------------------------------\n");
        printf(" Multiprocessor count:                %d\n", prop.multiProcessorCount);
        printf(" Shared memory per block:             %zu bytes\n", prop.sharedMemPerBlock);
        printf(" Registers per block:                 %d\n", prop.regsPerBlock);
        printf(" Threads in warp:                     %d\n", prop.warpSize);
        printf(" Max threads per block:               %d\n", prop.maxThreadsPerBlock);
        printf(" Max thread dimensions:               (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2]);
        printf(" Max grid dimensions:                 (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2]);
        printf(" ------------------------------------------------\n\n");
    }

    return 0;
}
