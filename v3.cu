#include <stdio.h>

#define SIDE 256

#include "float.c"
;
const int dsize = sizeof(float);

__device__
float Vz1(float x, float y, float xi, float nu, float z1, float z2, float H)
{
	float x_dif = (xi - x);
	float y_dif = (nu - y);
	float z_dif2 = (z2 - H);
	float z_dif1 = (z1 - H);

	float R1 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif1 * z_dif1);
	float R2 = sqrt(x_dif * x_dif + y_dif * y_dif + z_dif2 * z_dif2);

	return 
		(nu == y ? 0 : y_dif * log((x_dif + R2) / (x_dif + R1))) + 
		(xi == x ? 0 : x_dif * log((y_dif + R2) / (y_dif + R1))) -

		((z_dif2 == 0 ? 0 : z_dif2 * atan(x_dif * y_dif / (z_dif2 * R2))) -
		(z_dif1 == 0 ? 0 : z_dif1 * atan(x_dif * y_dif / (z_dif1 * R1))));
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
void Calculate(float xLL, float yLL, float xStep, float yStep, float* top, float* bottom, float* result)
{
	float x = xLL + xStep * blockIdx.x;
	float y = yLL + yStep * blockIdx.y;
	
	float x1 = xLL + xStep * threadIdx.x;
	float x2 = x1 + xStep;
	
	float y1 = yLL + yStep * threadIdx.y;
	float y2 = y1 + yStep;
	
	int pos_grid = threadIdx.x + threadIdx.y * SIDE;
	float t = top[pos_grid];
	float b = bottom[pos_grid];
	
	int pos_result = blockIdx.x + blockIdx.y * SIDE;
	
	float r = Vz3(x, y, x1, x2, y1, y2, t, b, 0);
	
	//atomicAdd(&result[pos_result], r);
	result[pos_result] += r;
}

int main()
{
	float *result = (float*)malloc(SIDE * SIDE * dsize);
	memset(result, 0, SIDE * SIDE * dsize);
	
	float *resultd, *catd, *zerod;
	cudaMalloc((void**)&resultd, SIDE * SIDE * dsize);
	cudaMalloc((void**)&catd, SIDE * SIDE * dsize);
	cudaMalloc((void**)&zerod, SIDE * SIDE * dsize);
	
	cudaMemcpy(zerod, result, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(resultd, result, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(catd, cat, SIDE * SIDE * dsize, cudaMemcpyHostToDevice);
	
	dim3 blocks(1, 1);
	dim3 threads(1, 1);
	
	//Vz3<<<blocks,threads>>>(0, 0, -100000000, 100000000, -100000000, 100000000, 1, 0, 0, result);
	//V<<<1,1>>>(result);
	Calculate<<<blocks,threads>>>(10000, 11000, 4, 4, zerod, catd, resultd);
	
	cudaMemcpy(result, resultd, SIDE * SIDE * dsize, cudaMemcpyDeviceToHost);
	cudaFree( resultd );
	printf("%f\n", result[0]);
	
	return 0;
}
