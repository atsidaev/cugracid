#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../global.h"

#define LINES_PER_BLOCK 1
#define THREAD_COUNT (SIDE / 16)

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
		(nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
		(xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

		((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
		(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1))));
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
void Calculate(int xMin, int yLine, FLOAT xLL, FLOAT yLL, FLOAT xStep, FLOAT yStep, FLOAT* top, FLOAT* bottom, FLOAT* result)
{
	__shared__ FLOAT sync[THREAD_COUNT];
	
	int xPos = xMin + threadIdx.x;// % (SIDE - xMin);
	int yPos = yLine;// + threadIdx.x / (SIDE - xMin);
	
	FLOAT x = xLL + xStep * blockIdx.x;
	FLOAT y = yLL + yStep * blockIdx.y;
	
	FLOAT x1 = xLL + xStep * xPos;
	FLOAT x2 = x1 + xStep;
	
	FLOAT y1 = yLL + yStep * yPos;
	FLOAT y2 = y1 + yStep;
	
	int pos_grid = xPos + yPos * SIDE;
	FLOAT t = 40.326; //top[pos_grid];
	FLOAT b = bottom[pos_grid];
	
	int pos_result = blockIdx.x + blockIdx.y * SIDE;
	
	FLOAT r = Vz3(x, y, x1, x2, y1, y2, t, b, 0);
	
	sync[threadIdx.x] = r;
	FLOAT res = result[pos_result];
	__syncthreads();
	if (threadIdx.x)
		return;

	for (int i = 0; i < THREAD_COUNT; i++)
		res += sync[i];
	result[pos_result] = res;
}

extern "C" { 

int CalculateVz(FLOAT* top, FLOAT* bottom, FLOAT* result)
{
	int returnCode = 1;
	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Found %d CUDA devices, ", deviceCount);
	// We need to get so many devices as we can to split data to equal-sized portions
	while (SIDE % deviceCount != 0)
		deviceCount--;
	printf("using %d of them\n", deviceCount);
	
	memset(result, 0, SIDE * SIDE * dsize);
	
	FLOAT *resultd[deviceCount], *bottomd[deviceCount], *topd[deviceCount];
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaMalloc((void**)&resultd[dev], SIDE * SIDE * dsize);
		cudaMalloc((void**)&bottomd[dev], SIDE * SIDE * dsize);
		cudaMalloc((void**)&topd[dev], SIDE * SIDE * dsize);

		cudaMemcpy(topd[dev], top, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(bottomd[dev], bottom, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(resultd[dev], result, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	}

	dim3 blocks(SIDE, SIDE);
	dim3 threads(THREAD_COUNT);

	for (int x = 0; x < SIDE; x += THREAD_COUNT)
	{
		for (int i = 0; i < SIDE; i+=LINES_PER_BLOCK * deviceCount)
		{
			for (int dev = 0; dev < deviceCount; dev++)
			{
				cudaSetDevice(dev);
				Calculate<<<blocks,threads>>>(x, i + LINES_PER_BLOCK * dev, 10017.376448317, 6395.193574, 3.0982365948353, 4.1303591058824, topd[dev], bottomd[dev], resultd[dev]);
			}
		}
	}
	
	FLOAT* result_tmp;
	cudaHostAlloc((void**)&result_tmp, SIDE * SIDE * dsize, cudaHostAllocDefault);
	
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
	
		cudaMemcpy(result_tmp, resultd[dev], SIDE * SIDE * dsize, cudaMemcpyDeviceToHost);
		for (int i = 0; i < SIDE * SIDE; i++)
			result[i] += result_tmp[i];
		cudaFree(resultd[dev]);
		cudaFree(topd[dev]);
		cudaFree(bottomd[dev]);
	}
	return returnCode;
}

}