#include <stdio.h>
#include <vector>

#ifdef __HIP__
#include "hip/hip_runtime.h"
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

void cudaPrintInfoForOneGpu(int num) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, num);
	printf("%s API version %d.%d\n", props.name, props.major, props.minor);
	printf("Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("Threads per block: %d\n", props.maxThreadsPerBlock);
	printf("Registers per block: %d\n", props.regsPerBlock);
}

void cudaPrintInfo() {
	cudaPrintInfoForOneGpu(0);
}

void cudaPrintInfo(std::vector<unsigned char> devices_list)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices, using ", deviceCount);
	if (devices_list.size() > 0)
		printf("%d", devices_list.size());
	else
		printf("all");
	printf(" of them\n");

	if (devices_list.size() > 0) {
		for (auto i = 0; i < devices_list.size(); i++)
			cudaPrintInfoForOneGpu(devices_list[i]);
	}
	else
		cudaPrintInfoForOneGpu(0);

	if (devices_list.size() == 0) {
		for (auto i = 0; i < deviceCount; i++)
			devices_list.push_back(i);
	}
}

