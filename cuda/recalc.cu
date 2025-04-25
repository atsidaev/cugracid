#include <stdio.h>
#include <math.h>

#include "hip/hip_runtime.h"

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"
#include "gpu_global.h"

#define sign(X) (X < 0 ? -1 : 1)

__device__
CUDA_FLOAT F(CUDA_FLOAT X, CUDA_FLOAT Y, CUDA_FLOAT H)
{
	return sign(X) * sign(Y) * (atan(abs(X * Y) / (H * sqrt(X * X + Y * Y + H * H))));
}

__device__
CUDA_FLOAT F1(CUDA_FLOAT X, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT H)
{
	return F(X, y2, H) - F(X, y1, H);
}

__device__
CUDA_FLOAT F2(CUDA_FLOAT x1, CUDA_FLOAT x2, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT H)
{
	return F1(x2, y1, y2, H) - F1(x1, y1, y2, H);
}

__global__
void RecalcCU(int first_block_pos, int nCol, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xStep, CUDA_FLOAT yStep, CUDA_FLOAT Height, CUDA_FLOAT* field, CUDA_FLOAT* result)
{
	__shared__ CUDA_FLOAT sync[THREADS_COUNT];

	// Point of calculation
	int pos_grid = (first_block_pos + threadIdx.x);
	int xPos = pos_grid % nCol;
	int yPos = pos_grid / nCol;
	
	CUDA_FLOAT x_ = xLL + xStep * xPos;
	CUDA_FLOAT y_ = yLL + yStep * yPos;
	
	CUDA_FLOAT x = xLL + xStep * blockIdx.x;
	CUDA_FLOAT y = yLL + yStep * blockIdx.y;
	
	CUDA_FLOAT x1 = x - xStep / 2;
	CUDA_FLOAT x2 = x + xStep / 2;
	
	CUDA_FLOAT y1 = y - yStep / 2;
	CUDA_FLOAT y2 = y + yStep / 2;

	int pos_result = blockIdx.x + blockIdx.y * nCol;
	
	CUDA_FLOAT U = field[pos_grid];
	CUDA_FLOAT r = U * F2(x1 - x_, x2 - x_, y1 - y_, y2 - y_, Height);
 	
	sync[threadIdx.x] = r;
	CUDA_FLOAT res = result[pos_result];
	__syncthreads();
	if (threadIdx.x)
		return;

	for (int i = 0; i < THREADS_COUNT; i++)
		res += sync[i];
	result[pos_result] = res;
}

int Recalc(CUDA_FLOAT* field, CUDA_FLOAT height, CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount)
{
	int returnCode = 1;
	
	int deviceCount;
	hipGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices, ", deviceCount);
	// We need to get so many devices as we can to split data to equal-sized portions
	while (nRow % deviceCount != 0)
		deviceCount--;
	printf("using %d of them\n", deviceCount);
	if (deviceCount == 0)
		return 0;
	
	memset(result, 0, nCol * nRow * dsize);
	
	CUDA_FLOAT **fieldd, **resultd;
	fieldd = new CUDA_FLOAT*[deviceCount];
	resultd = new CUDA_FLOAT*[deviceCount];

	for (int dev = 0; dev < deviceCount; dev++)
	{
		hipSetDevice(dev);
		hipMalloc((void**)&resultd[dev], nCol * nRow * dsize);
		hipMalloc((void**)&fieldd[dev], nCol * nRow * dsize);

		hipMemcpy(fieldd[dev], field, nCol * nRow * dsize, hipMemcpyHostToDevice);
		hipMemcpy(resultd[dev], result, nCol * nRow * dsize, hipMemcpyHostToDevice);
	}

	dim3 blocks(nCol, nRow);
	dim3 threads(THREADS_COUNT);

	for (int pos = firstRowToCalculate * nCol; pos < (firstRowToCalculate + rowsToCalculateCount) * nCol;)
	{
		for (int dev = 0; dev < deviceCount; dev++, pos += THREADS_COUNT)
		{
			hipSetDevice(dev);
			hipLaunchKernelGGL(RecalcCU, dim3(blocks), dim3(threads), 0, 0, pos, nCol, 10017.376448317, 6395.193574, 3.0982365948353, 4.1303591058824, height, fieldd[dev], resultd[dev]);
		}
	}

	
	CUDA_FLOAT* result_tmp;
	hipHostMalloc((void**)&result_tmp, nCol * nRow * dsize, hipHostMallocDefault);
	
	for (int dev = 0; dev < deviceCount; dev++)
	{
		hipSetDevice(dev);
		hipDeviceSynchronize();
		hipError_t error = hipGetLastError();
		if (error != hipSuccess)
		{
			printf("Error: %s\n", hipGetErrorString(error));
			returnCode = 0;
		}
	
		hipMemcpy(result_tmp, resultd[dev], nCol * nRow * dsize, hipMemcpyDeviceToHost);
		for (int i = 0; i < nCol * nRow; i++)
			result[i] += result_tmp[i];
		hipFree(resultd[dev]);
		hipFree(fieldd[dev]);
	}

	delete[] fieldd;
	delete[] resultd;

	return returnCode;
}
