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
