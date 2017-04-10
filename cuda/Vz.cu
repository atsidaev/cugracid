#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"

#define THREADS_COUNT 128

__device__
FLOAT Vz1(FLOAT x, FLOAT y, FLOAT xi, FLOAT nu, FLOAT z1, FLOAT z2, FLOAT H)
{
	FLOAT x_dif = (xi - x);
	FLOAT y_dif = (nu - y);
	FLOAT z_dif2 = (z2 - H);
	FLOAT z_dif1 = (z1 - H);

	FLOAT R1 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif1 * z_dif1);
	FLOAT R2 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif2 * z_dif2);

	return 
		-((nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
		  (xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

		((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
		(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1)))));
}

__device__
FLOAT Vz2(FLOAT x, FLOAT y, FLOAT xi, FLOAT y1, FLOAT y2, FLOAT z1, FLOAT z2, FLOAT H)
{
	return Vz1(x, y, xi, y2, z1, z2, H) - Vz1(x, y, xi, y1, z1, z2, H);
}

__device__
FLOAT Vz3(FLOAT x, FLOAT y, FLOAT x1, FLOAT x2, FLOAT y1, FLOAT y2, FLOAT z1, FLOAT z2, FLOAT H)
{
	return Vz2(x, y, x2, y1, y2, z1, z2, H) - Vz2(x, y, x1, y1, y2, z1, z2, H);
}

__global__
void Calculate(int first_block_pos, int maximumPos, int nCol, FLOAT xLL, FLOAT yLL, FLOAT xStep, FLOAT yStep, FLOAT* top, FLOAT* bottom, FLOAT* result)
{
	__shared__ FLOAT sync[THREADS_COUNT];
	
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
	
	FLOAT x = xLL + xStep * blockIdx.x + xStep / 2;
	FLOAT y = yLL + yStep * blockIdx.y + yStep / 2;
	
	FLOAT x1 = xLL + xStep * xPos;
	FLOAT x2 = x1 + xStep;
	
	FLOAT y1 = yLL + yStep * yPos;
	FLOAT y2 = y1 + yStep;

	FLOAT t = top[pos_grid];
	FLOAT b = bottom[pos_grid];
	
	int pos_result = blockIdx.x + blockIdx.y * nCol;
	
	FLOAT r = Vz3(x, y, x1, x2, y1, y2, t, b, 0);
	
	// printf("Field at (%f,%f) for (%f..%f,%f..%f,%f..%f) is %f\n", x,y,x1,x2,y1,y2,t,b,r);
	
	sync[threadIdx.x] = r;
	FLOAT res = result[pos_result];
	__syncthreads();
	if (threadIdx.x)
		return;

	for (int i = 0; i < THREADS_COUNT; i++)
		res += sync[i];
	result[pos_result] = res;
}

int CalculateVz(FLOAT* top, FLOAT* bottom, FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, FLOAT xLL, FLOAT yLL, FLOAT xSize, FLOAT ySize)
{
	int returnCode = 1;
	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices, ", deviceCount);
	if (deviceCount == 0)
		return 0;
	
	memset(result, 0, nCol * nRow * dsize);
	
	FLOAT **resultd, **bottomd, **topd;
	resultd = new FLOAT*[deviceCount];
	bottomd = new FLOAT*[deviceCount];
	topd = new FLOAT*[deviceCount];

	// Setup inbound and outbound arrays for all CUDA devices
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaMalloc((void**)&resultd[dev], nCol * nRow * dsize);
		cudaMalloc((void**)&bottomd[dev], nCol * nRow * dsize);
		cudaMalloc((void**)&topd[dev], nCol * nRow * dsize);

		cudaMemcpy(topd[dev], top, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(bottomd[dev], bottom, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(resultd[dev], result, nCol * nRow * dsize, cudaMemcpyHostToDevice);
	}

	dim3 blocks(nCol, nRow);
	dim3 threads(THREADS_COUNT);

	int pos = firstRowToCalculate * nCol;
	int maximumPos = (firstRowToCalculate + rowsToCalculateCount) * nCol;
	int currentDevice = 0;
	while (pos < maximumPos)
	{
		cudaSetDevice(currentDevice);
		Calculate<<<blocks,threads>>>(pos, maximumPos, nCol, xLL, yLL, xSize, ySize, topd[currentDevice], bottomd[currentDevice], resultd[currentDevice]);
		pos += THREADS_COUNT;
		currentDevice = (currentDevice + 1) % deviceCount;
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
		cudaFree(topd[dev]);
		cudaFree(bottomd[dev]);
	}

	delete[] resultd;
	delete[] topd;
	delete[] bottomd;

	return returnCode;
}
