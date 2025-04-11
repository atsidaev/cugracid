#include <stdio.h>
#include <vector>
#include <math.h>

#ifdef _WIN32
#include <memory.h>
#endif

#define THREADS_COUNT 128

#include "../global.h"
#include "prism_Vz.cuh"

#ifdef __HIP__
#include "hipified.h"
#define LAUNCH_FUNCTION_ON_PRISM(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) hipLaunchKernelGGL(method, dim3(blocks), dim3(threads), 0, 0, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LAUNCH_FUNCTION_ON_PRISM(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) method<<<blocks,threads>>>(first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#endif

//#define SIMPLIFY_SOURCES

// T must contain calc(x, y, z, x1, x2, y1, y2, z1, z2)
// x, y, z - observation point
// x1..x2, y1..y2, z1..z2 - the prism

template<typename T>
__global__
void CalculateFromPrismOnGrid(
	/* Portion for calculaton for the parallel execution */ int first_block_pos, int maximumPos,
	/* Grid definition (both input grids and output grid must have the same params) */ int nCol, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xStep, CUDA_FLOAT yStep,
	CUDA_FLOAT* top, CUDA_FLOAT* bottom,
	/* Multiplier to be applied to the result if needed (NULL if not needed) */ CUDA_FLOAT* multiplier,
	CUDA_FLOAT* result)
{
	__shared__ CUDA_FLOAT sync[THREADS_COUNT];
	
	int pos_grid = (first_block_pos + threadIdx.x);
	if (pos_grid >= maximumPos)
	{
		// if we are outside of data which should be calculated by our block
		// then we need to skip this thread to avoid double calculation of the nodes
		sync[threadIdx.x] = 0;
		return;
	}

	int xPos = pos_grid % nCol;
	int yPos = pos_grid / nCol;
	
	CUDA_FLOAT x = xLL + xStep * blockIdx.x + xStep / 2;
	CUDA_FLOAT y = yLL + yStep * blockIdx.y + yStep / 2;
	
	CUDA_FLOAT x1 = xLL + xStep * xPos;
	CUDA_FLOAT x2 = x1 + xStep;
	
	CUDA_FLOAT y1 = yLL + yStep * yPos;
	CUDA_FLOAT y2 = y1 + yStep;

	CUDA_FLOAT t = top[pos_grid];
	CUDA_FLOAT b = bottom[pos_grid];
	
	int pos_result = blockIdx.x + blockIdx.y * nCol;
	
	CUDA_FLOAT r = T::calc(x, y, 0, x1, x2, y1, y2, t, b);
	if (multiplier != NULL)
		r *= multiplier[pos_grid];
	// printf("Field at (%f,%f) for (%f..%f,%f..%f,%f..%f) is %f\n", x,y,x1,x2,y1,y2,t,b,r);
	
	sync[threadIdx.x] = r;
	CUDA_FLOAT res = result[pos_result];
	__syncthreads();
	if (threadIdx.x)
		return;

	for (int i = 0; i < THREADS_COUNT; i++)
		res += sync[i];
	result[pos_result] = res;
}

void CalculateVzInParallel(dim3 blocks, dim3 threads, int first_block_pos, int maximumPos, int nCol, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xStep, CUDA_FLOAT yStep, CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result)
{
	LAUNCH_FUNCTION_ON_PRISM(
#ifdef SIMPLIFY_SOURCES
		CalculateFromPrismOnGrid<VzCalcSimplified>,
#else
		CalculateFromPrismOnGrid<VzCalc>,
#endif
		blocks,
		threads,
		first_block_pos, maximumPos, 
		nCol, xLL, yLL, xStep, yStep,
		top,bottom,
		dsigma,
		result);
}

int CalculateVz(CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xSize, CUDA_FLOAT ySize, std::vector<unsigned char> devices_list)
{
	int returnCode = 1;
	
	int deviceCount = devices_list.size();

	if (deviceCount == 0)
	{
		fprintf(stderr, "No CUDA devices found\n");
		return 0;
	}
	
	memset(result, 0, nCol * nRow * dsize);
	
	CUDA_FLOAT **resultd, **bottomd, **topd, **dsigmad;
	resultd = new CUDA_FLOAT*[deviceCount];
	bottomd = new CUDA_FLOAT*[deviceCount];
	topd = new CUDA_FLOAT*[deviceCount];
	dsigmad = new CUDA_FLOAT*[deviceCount];

	// Setup inbound and outbound arrays for all CUDA devices
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(devices_list[dev]);
		cudaMalloc((void**)&resultd[dev], nCol * nRow * dsize);
		cudaMalloc((void**)&bottomd[dev], nCol * nRow * dsize);
		cudaMalloc((void**)&topd[dev], nCol * nRow * dsize);

		cudaMemcpy(topd[dev], top, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(bottomd[dev], bottom, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(resultd[dev], result, nCol * nRow * dsize, cudaMemcpyHostToDevice);

		if (dsigma != NULL)
		{
			cudaMalloc((void**)&dsigmad[dev], nCol * nRow * dsize);
			cudaMemcpy(dsigmad[dev], dsigma, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		}
		else
		{
			dsigmad[dev] = NULL;
		}
	}

	dim3 blocks(nCol, nRow);
	dim3 threads(THREADS_COUNT);

	int pos = firstRowToCalculate * nCol;
	int maximumPos = (firstRowToCalculate + rowsToCalculateCount) * nCol;
	int currentDevice = 0;
	while (pos < maximumPos)
	{
		cudaSetDevice(devices_list[currentDevice]);
		CalculateVzInParallel(blocks, threads, pos, maximumPos, nCol, xLL, yLL, xSize, ySize, topd[currentDevice], bottomd[currentDevice], dsigmad[currentDevice], resultd[currentDevice]);
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
	
		cudaMemcpy(result_tmp, resultd[dev], nCol * nRow * dsize, cudaMemcpyDeviceToHost);
		for (int i = 0; i < nCol * nRow; i++)
			result[i] += result_tmp[i];
		cudaFree(resultd[dev]);
		cudaFree(topd[dev]);
		cudaFree(bottomd[dev]);
	}

	delete[] resultd;
	delete[] topd;
	delete[] bottomd;

	return returnCode;
}
