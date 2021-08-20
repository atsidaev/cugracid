#include <stdio.h>
#include <vector>
#include <math.h>

#ifdef __HIP__
#include "hip/hip_runtime.h"
#define cudaSetDevice hipSetDevice
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaFree hipFree
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define LAUNCH(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) hipLaunchKernelGGL(method, dim3(blocks), dim3(threads), 0, 0, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LAUNCH(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) method<<<blocks,threads>>>(first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#endif

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"

#define THREADS_COUNT 128

__device__
float Vz1(CUDA_FLOAT _x, CUDA_FLOAT _y, CUDA_FLOAT _xi, CUDA_FLOAT _nu, CUDA_FLOAT _z1, CUDA_FLOAT _z2, CUDA_FLOAT _H)
{
	float x = _x;
	float y = _y;
	float xi = _xi;
	float nu = _nu;
	float z1 = _z1;
	float z2 = _z2;
	float H = _H;

	float x_dif = (xi - x);
	float y_dif = (nu - y);
	float z_dif2 = (z2 - H);
	float z_dif1 = (z1 - H);

	float R1 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif1 * z_dif1);
	float R2 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif2 * z_dif2);

	return 
		-((nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
		  (xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

		((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
		(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1)))));
}

__device__
float Vz2(float x, float y, float xi, float y1, float y2, float z1, float z2, float H)
{
	return Vz1(x, y, xi, y2, z1, z2, H) - Vz1(x, y, xi, y1, z1, z2, H);
}

__device__
float Vz3(float x, float y, float x1, float x2, float y1, float y2, float z1, float z2, float H)
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
		LAUNCH(Calculate, blocks, threads, pos, maximumPos, nCol, xLL, yLL, xSize, ySize, topd[currentDevice], bottomd[currentDevice], dsigmad[currentDevice], resultd[currentDevice]);
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
