#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

#define SIDE 256
#define FLOAT double
const int dsize = sizeof(FLOAT);

#include "d:\\IGF\\2013-12-04\\fund.c"

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
void Calculate(int yLine, FLOAT xLL, FLOAT yLL, FLOAT xStep, FLOAT yStep, FLOAT* top, FLOAT* bottom, FLOAT* result, FLOAT* sync)
{
	FLOAT x = xLL + xStep * blockIdx.x;
	FLOAT y = yLL + yStep * blockIdx.y;
	
	FLOAT x1 = xLL + xStep * threadIdx.x;
	FLOAT x2 = x1 + xStep;
	
	FLOAT y1 = yLL + yStep * yLine;
	FLOAT y2 = y1 + yStep;
	
	int pos_grid = threadIdx.x + yLine * SIDE;
	FLOAT t = 40.326; //top[pos_grid];
	FLOAT b = bottom[pos_grid];
	
	int pos_result = blockIdx.x + blockIdx.y * SIDE;
	
	FLOAT r = Vz3(x, y, x1, x2, y1, y2, t, b, 0);
	
	//atomicAdd(&result[pos_result], r);
	//result[pos_result] += r;
	sync[pos_result * SIDE + threadIdx.x] = r;
	FLOAT res = result[pos_result];
	__syncthreads();
	for (int i = 0, p = pos_result * SIDE; i < SIDE; i++, p++)
		res += sync[p];
	result[pos_result] = res;
}

int main()
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	printf("%s API version %d.%d\n", props.name, props.major, props.minor);
	printf("Maximum texture dimensions: 1D: %d, 2D: %d, 3D: %d\n", props.maxTexture1D, props.maxTexture2D, props.maxTexture3D);
    
	printf("Max block dimensions: [%d, %d, %d]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
	printf("Threads per block: %d\n", props.maxThreadsPerBlock);
	
	FLOAT *result = (FLOAT*)malloc(SIDE * SIDE * dsize);
	memset(result, 0, SIDE * SIDE * dsize);

	FLOAT *sync= (FLOAT*)malloc(SIDE * SIDE * SIDE * dsize);
	memset(sync, 0, SIDE * SIDE * SIDE * dsize);

	FLOAT *resultd, *bottomd, *topd, *syncd;
	cudaMalloc((void**)&resultd, SIDE * SIDE * dsize);
	cudaMalloc((void**)&bottomd, SIDE * SIDE * dsize);
	cudaMalloc((void**)&topd, SIDE * SIDE * dsize);
	cudaMalloc((void**)&syncd, SIDE * SIDE * SIDE * dsize);

	cudaMemcpy(topd, result, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(resultd, result, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(bottomd, grid, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	
	dim3 blocks(SIDE, SIDE);
	dim3 threads(SIDE);
	
	//Vz3<<<blocks,threads>>>(0, 0, -100000000, 100000000, -100000000, 100000000, 1, 0, 0, result);
	//V<<<1,1>>>(result);
	for (int i = 0; i < SIDE; i++)
	{
		cudaMemset(syncd, 0, SIDE * SIDE * SIDE * dsize);
		Calculate<<<blocks,threads>>>(i, 10017.376448317, 6395.193574, 3.0982365948353, 4.1303591058824, topd, bottomd, resultd, syncd);
	}
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	printf("Error: %s\n", cudaGetErrorString(error));
	
	cudaMemcpy(result, resultd, SIDE * SIDE * dsize, cudaMemcpyDeviceToHost);
	cudaFree( resultd );
	printf("%f\n", result[(SIDE / 2) * SIDE + SIDE / 2]);
	
	FILE *f = fopen("d:\\fund_f.bin", "wb");
	fwrite(result, dsize, SIDE * SIDE, f);
	fclose(f);

	return 0;
}
