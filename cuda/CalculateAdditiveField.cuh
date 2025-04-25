#ifndef CALCULATE_ADDITIVE_FIELD_CUH
#define CALCULATE_ADDITIVE_FIELD_CUH

#include <vector>
#include "AbstractDeviceContext.cuh"

int CalculateAdditiveField(CUDA_FLOAT* result, int nCol, int nRow, int firstRowToCalculate, int rowsToCalculateCount, AbstractDeviceContext** deviceContexts, std::vector<unsigned char> devices_list);

#endif