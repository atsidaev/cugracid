#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" {

void cudaPrintInfo()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices\n", deviceCount);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	printf("%s API version %d.%d\n", props.name, props.major, props.minor);
	printf("Maximum texture dimensions: 1D: %d, 2D: %d, 3D: %d\n", props.maxTexture1D, props.maxTexture2D, props.maxTexture3D);
    
	printf("Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("Threads per block: %d\n", props.maxThreadsPerBlock);
	printf("Registers per block: %d\n", props.regsPerBlock);
}

}