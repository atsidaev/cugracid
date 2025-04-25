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

void initializeDevicesList(std::vector<unsigned char>& devices_list)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	fprintf(stderr, "Found %d GPU devices\n", deviceCount);
	if (devices_list.size() == 0) {
		for (auto i = 0; i < deviceCount; i++)
			devices_list.push_back(i);
	}
}

std::vector<unsigned char> getGpuDevices()
{
	std::vector<unsigned char> devices_list;
	initializeDevicesList(devices_list);
	return devices_list;
}

void cudaPrintInfoForOneGpu(int num) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, num);
	printf("%d: %s API version %d.%d\n", num, props.name, props.major, props.minor);
	printf("\tMax block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("\tThreads per block: %d\n", props.maxThreadsPerBlock);
	printf("\tRegisters per block: %d\n", props.regsPerBlock);
}

void cudaPrintInfo() {
	cudaPrintInfoForOneGpu(0);
}

void cudaPrintInfo(std::vector<unsigned char>& devices_list)
{
	fprintf(stderr, "Using %ld GPU devices\n", devices_list.size());

	if (devices_list.size() > 0) {
		for (auto i = 0; i < devices_list.size(); i++)
			cudaPrintInfoForOneGpu(devices_list[i]);
	}
	else
		cudaPrintInfoForOneGpu(0);
}

