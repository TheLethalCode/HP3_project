#include <cuda.h>
#include <cuda_runtime.h>

#define coarsen_factor 8
#define stride 32


__global__ void APSP_kernel1(int *dis, int k, int n) {
    int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
    int idy0 = threadIdx.y + blockIdx.y * blockDim.y;

    int idx1 = idx0 + stride;
    int idy1 = idy0 + stride;

    int idx2 = idx1 + stride;
    int idy2 = idy1 + stride;

    int idx3 = idx2 + stride;
    int idy3 = idy2 + stride;

    int idx4 = idx3 + stride;
    int idy4 = idy3 + stride;

    int idx5 = idx4 + stride;
    int idy5 = idy4 + stride;

    int idx6 = idx5 + stride;
    int idy6 = idy5 + stride;

    int idx7 = idx6 + stride;
    int idy7 = idy6 + stride;
    
    if (idx0 < n && idy0 < n) {
        dis[idx0*n + idy0] = min(dis[idx0*n + idy0], dis[idx0*n + k] + dis[k*n + idy0]);
    }

    if (idx1 < n && idy1 < n) {
        dis[idx1*n + idy1] = min(dis[idx1*n + idy1], dis[idx1*n + k] + dis[k*n + idy1]);
    }

    if (idx2 < n && idy2 < n) {
        dis[idx2*n + idy2] = min(dis[idx2*n + idy2], dis[idx2*n + k] + dis[k*n + idy2]);
    }

    if (idx3 < n && idy3 < n) {
        dis[idx3*n + idy3] = min(dis[idx3*n + idy3], dis[idx3*n + k] + dis[k*n + idy3]);
    }

    if (idx4 < n && idy4 < n) {
        dis[idx4*n + idy3] = min(dis[idx4*n + idy3], dis[idx4*n + k] + dis[k*n + idy4]);
    }
    
    if (idx5 < n && idy5 < n) {
        dis[idx5*n + idy5] = min(dis[idx5*n + idy4], dis[idx5*n + k] + dis[k*n + idy5]);
    }
    
    if (idx6 < n && idy6 < n) {
        dis[idx6*n + idy6] = min(dis[idx6*n + idy6], dis[idx6*n + k] + dis[k*n + idy6]);
    }
    
    if (idx7 < n && idy7 < n) {
        dis[idx7*n + idy7] = min(dis[idx7*n + idy7], dis[idx7*n + k] + dis[k*n + idy7]);
    }

}