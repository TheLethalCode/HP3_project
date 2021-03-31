#include <cuda.h>
#include <cuda_runtime.h>

#define coarsen_factor 2
#define stride 32


__global__ void APSP_kernel1(int *dis, int k, int n) {
    int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
    int idy0 = threadIdx.y + blockIdx.y * blockDim.y;

    int idx1 = idx0 + stride;
    int idy1 = idy0 + stride;

    if (idx0 < n && idy0 < n) {
        dis[idx0*n + idy0] = min(dis[idx0*n + idy0], dis[idx0*n + k] + dis[k*n + idy0]);
    }

    if (idx1 < n && idy1 < n) {
        dis[idx1*n + idy1] = min(dis[idx1*n + idy1], dis[idx1*n + k] + dis[k*n + idy1]);
    }
}