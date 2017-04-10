#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"

#define THREADS_COUNT 128

#define sign(X) (X < 0 ? -1 : 1)

__device__
FLOAT F(FLOAT X, FLOAT Y, FLOAT H)
{
	return sign(X) * sign(Y) * (atan(abs(X * Y) / (H * sqrt(X * X + Y * Y + H * H))));
}

__device__
FLOAT F1(FLOAT X, FLOAT y1, FLOAT y2, FLOAT H)
{
	return F(X, y2, H) - F(X, y1, H);
}

__device__
FLOAT F2(FLOAT x1, FLOAT x2, FLOAT y1, FLOAT y2, FLOAT H)
{
	return F1(x2, y1, y2, H) - F1(x1, y1, y2, H);
}

__global__
void RecalcCU(int first_block_pos, int nCol, FLOAT xLL, FLOAT yLL, FLOAT xStep, FLOAT yStep, FLOAT Height, FLOAT* field, FLOAT* result)
{
	__shared__ FLOAT sync[THREADS_COUNT];

	// Point of calculation
	int pos_grid = (first_block_pos + threadIdx.x);
	int xPos = pos_grid % nCol;
	int yPos = pos_grid / nCol;
	
	FLOAT x_ = xLL + xStep * xPos;
	FLOAT y_ = yLL + yStep * yPos;
	
	FLOAT x = xLL + xStep * blockIdx.x;
	FLOAT y = yLL + yStep * blockIdx.y;
	
	FLOAT x1 = x - xStep / 2;
	FLOAT x2 = x + xStep / 2;
	
	FLOAT y1 = y - yStep / 2;
	FLOAT y2 = y + yStep / 2;

	int pos_result = blockIdx.x + blockIdx.y * nCol;
	
	FLOAT U = field[pos_grid];
	FLOAT r = U * F2(x1 - x_, x2 - x_, y1 - y_, y2 - y_, Height);
 	
	sync[threadIdx.x] = r;
	FLOAT res = result[pos_result];
	__syncthreads();
	if (threadIdx.x)
		return;

	for (int i = 0; i < THREADS_COUNT; i++)
		res += sync[i];
	result[pos_result] = res;
}

int Recalc(FLOAT* field, FLOAT height, FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount)
{
	int returnCode = 1;
	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices, ", deviceCount);
	// We need to get so many devices as we can to split data to equal-sized portions
	while (nRow % deviceCount != 0)
		deviceCount--;
	printf("using %d of them\n", deviceCount);
	if (deviceCount == 0)
		return 0;
	
	memset(result, 0, nCol * nRow * dsize);
	
	FLOAT **fieldd, **resultd;
	fieldd = new FLOAT*[deviceCount];
	resultd = new FLOAT*[deviceCount];

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaMalloc((void**)&resultd[dev], nCol * nRow * dsize);
		cudaMalloc((void**)&fieldd[dev], nCol * nRow * dsize);

		cudaMemcpy(fieldd[dev], field, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(resultd[dev], result, nCol * nRow * dsize, cudaMemcpyHostToDevice);
	}

	dim3 blocks(nCol, nRow);
	dim3 threads(THREADS_COUNT);

	for (int pos = firstRowToCalculate * nCol; pos < (firstRowToCalculate + rowsToCalculateCount) * nCol;)
	{
		for (int dev = 0; dev < deviceCount; dev++, pos += THREADS_COUNT)
		{
			cudaSetDevice(dev);
			RecalcCU<<<blocks,threads>>>(pos, nCol, 10017.376448317, 6395.193574, 3.0982365948353, 4.1303591058824, height, fieldd[dev], resultd[dev]);
		}
	}

	
	FLOAT* result_tmp;
	cudaHostAlloc((void**)&result_tmp, nCol * nRow * dsize, cudaHostAllocDefault);
	
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
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
		cudaFree(fieldd[dev]);
	}

	delete[] fieldd;
	delete[] resultd;

	return returnCode;
}
