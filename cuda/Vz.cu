#include <stdio.h>
#include <math.h>

#include "hip/hip_runtime.h"

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"

#define THREADS_COUNT 128

__device__
CUDA_FLOAT Vz1(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT xi, CUDA_FLOAT nu, CUDA_FLOAT z1, CUDA_FLOAT z2, CUDA_FLOAT H)
{
	CUDA_FLOAT x_dif = (xi - x);
	CUDA_FLOAT y_dif = (nu - y);
	CUDA_FLOAT z_dif2 = (z2 - H);
	CUDA_FLOAT z_dif1 = (z1 - H);

	CUDA_FLOAT R1 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif1 * z_dif1);
	CUDA_FLOAT R2 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif2 * z_dif2);

	return 
		-((nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
		  (xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

		((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
		(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1)))));
}

__device__
CUDA_FLOAT Vz2(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT xi, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT z1, CUDA_FLOAT z2, CUDA_FLOAT H)
{
	return Vz1(x, y, xi, y2, z1, z2, H) - Vz1(x, y, xi, y1, z1, z2, H);
}

__device__
CUDA_FLOAT Vz3(CUDA_FLOAT x, CUDA_FLOAT y, CUDA_FLOAT x1, CUDA_FLOAT x2, CUDA_FLOAT y1, CUDA_FLOAT y2, CUDA_FLOAT z1, CUDA_FLOAT z2, CUDA_FLOAT H)
{
	return Vz2(x, y, x2, y1, y2, z1, z2, H) - Vz2(x, y, x1, y1, y2, z1, z2, H);
}

__global__
void Calculate(int first_block_pos, int maximumPos, int nCol, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xStep, CUDA_FLOAT yStep, CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result)
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
	
	CUDA_FLOAT r = Vz3(x, y, x1, x2, y1, y2, t, b, 0);
	if (dsigma != NULL)
		r *= dsigma[pos_grid];
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

int CalculateVz(CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xSize, CUDA_FLOAT ySize)
{
	int returnCode = 1;
	
	int deviceCount;
	hipGetDeviceCount(&deviceCount);

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
		hipSetDevice(dev);
		hipMalloc((void**)&resultd[dev], nCol * nRow * dsize);
		hipMalloc((void**)&bottomd[dev], nCol * nRow * dsize);
		hipMalloc((void**)&topd[dev], nCol * nRow * dsize);

		hipMemcpy(topd[dev], top, nCol * nRow * dsize, hipMemcpyHostToDevice);
		hipMemcpy(bottomd[dev], bottom, nCol * nRow * dsize, hipMemcpyHostToDevice);
		hipMemcpy(resultd[dev], result, nCol * nRow * dsize, hipMemcpyHostToDevice);

		if (dsigma != NULL)
		{
			hipMalloc((void**)&dsigmad[dev], nCol * nRow * dsize);
			hipMemcpy(dsigmad[dev], dsigma, nCol * nRow * dsize, hipMemcpyHostToDevice);
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
		hipSetDevice(currentDevice);
		hipLaunchKernelGGL(Calculate, dim3(blocks), dim3(threads), 0, 0, pos, maximumPos, nCol, xLL, yLL, xSize, ySize, topd[currentDevice], bottomd[currentDevice], dsigmad[currentDevice], resultd[currentDevice]);
		pos += THREADS_COUNT;
		currentDevice = (currentDevice + 1) % deviceCount;
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
		hipFree(topd[dev]);
		hipFree(bottomd[dev]);
	}

	delete[] resultd;
	delete[] topd;
	delete[] bottomd;

	return returnCode;
}
