#include <cuda.h>
#include <cuda_runtime.h>

__global__ void APSP_kernel1(int *dis, int k, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if (idx < n && idy < n) {
        dis[idx*n + idy] = min(dis[idx*n + idy], dis[idx*n + k] + dis[k*n + idy]);
    }
}