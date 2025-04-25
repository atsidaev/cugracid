#ifndef CALCULATE_ADDITIVE_FIELD_CUH
#define CALCULATE_ADDITIVE_FIELD_CUH

#include <stdio.h>
#include <vector>
#include "AbstractDeviceContext.cuh"

template<typename T>
int CalculateAdditiveField(CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, std::vector<T*> deviceContexts, std::vector<unsigned char> devices_list)
{
	int returnCode = 1;

	int deviceCount = devices_list.size();

	if (deviceCount == 0)
	{
		fprintf(stderr, "No CUDA devices found\n");
		return 0;
	}
	
	memset(result, 0, nCol * nRow * dsize);

	dim3 blocks(nCol, nRow);
	dim3 threads(THREADS_COUNT);

	int pos = firstRowToCalculate * nCol;
	int maximumPos = (firstRowToCalculate + rowsToCalculateCount) * nCol;
	int currentDevice = 0;
	while (pos < maximumPos)
	{
		deviceContexts.at(currentDevice)->RunCalculation(blocks, threads, pos, maximumPos);
		pos += THREADS_COUNT;
		currentDevice = (currentDevice + 1) % deviceCount;
	}

	CUDA_FLOAT* result_tmp;
	cudaHostAlloc((void**)&result_tmp, nCol * nRow * dsize, cudaHostAllocDefault);
	
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(devices_list[dev]);
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			printf("Error: %s\n", cudaGetErrorString(error));
			returnCode = 0;
		}
	
		deviceContexts.at(dev)->GetResult(result_tmp);

		// Accumulate result. Works for additive fields only!
		for (int i = 0; i < nCol * nRow; i++)
			result[i] += result_tmp[i];
	}

	return returnCode;
}

#endif