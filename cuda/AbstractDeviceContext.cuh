#ifndef BASE_DEVICE_CONTEXT_H
#define BASE_DEVICE_CONTEXT_H

#include "../global.h"

class AbstractDeviceContext {
public:
	virtual void RunCalculation(dim3 blocks, dim3 threads, int pos, int maximumPos) = 0;
	virtual void GetResult(CUDA_FLOAT* buffer) = 0;
};

#endif