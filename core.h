#ifndef CUDA_CORE
#define CUDA_CORE
#include <cuda.h>
#include <cuda_runtime.h>

bool cudaCheck(cudaError_t);
void DeviceProp();

cudaEvent_t start, stop;

#endif