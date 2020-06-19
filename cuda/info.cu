#include <stdio.h>

#include "hip/hip_runtime.h"

void cudaPrintInfo()
{
	int deviceCount;
	hipGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices\n", deviceCount);

	hipDeviceProp_t props;
	hipGetDeviceProperties(&props, 0);
	printf("%s API version %d.%d\n", props.name, props.major, props.minor);
	printf("Maximum texture dimensions: 1D: %d, 2D: %d, 3D: %d\n", props.maxTexture1D, props.maxTexture2D, props.maxTexture3D);
    
	printf("Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("Threads per block: %d\n", props.maxThreadsPerBlock);
	printf("Registers per block: %d\n", props.regsPerBlock);
}

