#include <stdio.h>
#include <vector>
#include "hip/hip_runtime.h"

void cudaPrintInfoForOneGpu(int num) {
	hipDeviceProp_t props;
	hipGetDeviceProperties(&props, num);
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
	hipGetDeviceCount(&deviceCount);
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

