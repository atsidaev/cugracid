#include <stdio.h>

#define SIDE 256

#include "float.c"

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

__global__
void Vz3(float x, float y, float x1, float x2, float y1, float y2, float z1, float z2, float H, float* result)
{
	*result = Vz2(x, y, x2, y1, y2, z1, z2, H) - Vz2(x, y, x1, y1, y2, z1, z2, H);
}

int main()
{
	float *result;
	cudaMalloc((void**)&result, dsize);
	
	dim3 blocks(SIDE, SIDE);
	dim3 threads(SIDE, SIDE);
	
	Vz3<<<blocks,threads>>>(0, 0, -100000000, 100000000, -100000000, 100000000, 1, 0, 0, result);
	//V<<<1,1>>>(result);
	
	float r;
	cudaMemcpy(&r, result, dsize, cudaMemcpyDeviceToHost );
	cudaFree( result );
	printf("%f\n", r);
	
	return 0;
}
