#include "../global.h"

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