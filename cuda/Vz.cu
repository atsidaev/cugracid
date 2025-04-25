#include <vector>
#include <math.h>

#ifdef _WIN32
#include <memory.h>
#endif

#include "../global.h"
#include "prism_Vz.cuh"

#include "VzCalculationDeviceContext.cuh"
#include "CalculateAdditiveField.cuh"

//#define SIMPLIFY_SOURCES
#ifdef SIMPLIFY_SOURCES
	#define Calc VzCalcSimplified
#else
	#define Calc VzCalc
#endif

int CalculateVz(CUDA_FLOAT* top, CUDA_FLOAT* bottom, CUDA_FLOAT* dsigma, CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, CUDA_FLOAT xLL, CUDA_FLOAT yLL, CUDA_FLOAT xSize, CUDA_FLOAT ySize, std::vector<unsigned char> devices_list)
{
	int deviceCount = devices_list.size();
	VzCalculationDeviceContext<Calc>* deviceContexts[deviceCount];

	// Setup inbound and outbound arrays for all CUDA devices
	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(devices_list[dev]);
		deviceContexts[dev] = new VzCalculationDeviceContext<Calc>(top, bottom, dsigma, result, nCol, nRow, xLL, yLL, xSize, ySize);
	}

	return CalculateAdditiveField(result, nCol, nRow, firstRowToCalculate, rowsToCalculateCount, reinterpret_cast<AbstractDeviceContext**>(deviceContexts), devices_list);
}
