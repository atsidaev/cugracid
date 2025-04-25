#include "../global.h"
#include "gpu_global.h"
#include "AbstractDeviceContext.cuh"
#include "CalculateFromPrismOnGrid.cuh"

#ifdef __HIP__
#include "hipified.h"
#define LAUNCH_FUNCTION_ON_PRISM(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) hipLaunchKernelGGL(method, dim3(blocks), dim3(threads), 0, 0, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define LAUNCH_FUNCTION_ON_PRISM(method, blocks, threads, first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result) method<<<blocks,threads>>>(first_block_pos, maximumPos, nCol, xLL, yLL, xStep, yStep, top, bottom, dsigma, result)
#endif

template<typename T>
void CalculateVzInParallel(dim3 blocks, dim3 threads, int first_block_pos, int maximumPos, int nCol, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xStep, CUDA_FLOAT yStep, CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result)
{
	LAUNCH_FUNCTION_ON_PRISM(
		CalculateFromPrismOnGrid<T>,
		blocks,
		threads,
		first_block_pos, maximumPos, 
		nCol, xLL, yLL, xStep, yStep,
		top,bottom,
		dsigma,
		result);
}

template<typename T>
class VzCalculationDeviceContext : AbstractDeviceContext {
private:
	int _device;
	CUDA_FLOAT *resultd, *bottomd, *topd, *dsigmad;
	int _nCol, _nRow;
	double _xLL, _yLL, _xSize, _ySize;
public:
	VzCalculationDeviceContext(int dev, CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result, int nCol, int nRow, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xSize, CUDA_FLOAT ySize) {
		this->_nCol = nCol;
		this->_nRow = nRow;

		this->_xLL = xLL;
		this->_yLL = yLL;
		this->_xSize = xSize;
		this->_ySize = ySize;

		this->_device = dev;
		cudaSetDevice(dev);

		cudaMalloc((void**)&this->resultd, nCol * nRow * dsize);
		cudaMalloc((void**)&this->bottomd, nCol * nRow * dsize);
		cudaMalloc((void**)&this->topd, nCol * nRow * dsize);
		
		cudaMemcpy(this->topd, top, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(this->bottomd, bottom, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		cudaMemcpy(this->resultd, result, nCol * nRow * dsize, cudaMemcpyHostToDevice);

		if (dsigma != NULL)
		{
			cudaMalloc((void**)&this->dsigmad, nCol * nRow * dsize);
			cudaMemcpy(this->dsigmad, dsigma, nCol * nRow * dsize, cudaMemcpyHostToDevice);
		}
		else
		{
			this->dsigmad = NULL;
		}
	}

	void RunCalculation(dim3 blocks, dim3 threads, int pos, int maximumPos) {
		cudaSetDevice(_device);
		CalculateVzInParallel<T>(blocks, threads, pos, maximumPos, _nCol, _xLL, _yLL, _xSize, _ySize, topd, bottomd, dsigmad, resultd);
	}

	void GetResult(CUDA_FLOAT* buffer) {
		cudaSetDevice(_device);
		cudaMemcpy(buffer, this->resultd, this->_nCol * this->_nRow * dsize, cudaMemcpyDeviceToHost);
	}

	~VzCalculationDeviceContext() {
		cudaFree(this->resultd);
		cudaFree(this->topd);
		cudaFree(this->bottomd);
	}
};