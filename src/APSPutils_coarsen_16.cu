#include <cuda.h>
#include <cuda_runtime.h>

#define coarsen_factor 16
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

    int idx8 = idx7 + stride;
    int idy8 = idy7 + stride;

    int idx9 = idx8 + stride;
    int idy9 = idy8 + stride;

    int idx10 = idx9 + stride;
    int idy10 = idy9 + stride;

    int idx11 = idx10 + stride;
    int idy11 = idy10 + stride;

    int idx12 = idx11 + stride;
    int idy12 = idy11 + stride;

    int idx13 = idx12 + stride;
    int idy13 = idy12 + stride;

    int idx14 = idx13 + stride;
    int idy14 = idy13 + stride;

    int idx15 = idx14 + stride;
    int idy15 = idy14 + stride;


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
        dis[idx4*n + idy4] = min(dis[idx4*n + idy4], dis[idx4*n + k] + dis[k*n + idy4]);
    }
    
    if (idx5 < n && idy5 < n) {
        dis[idx5*n + idy5] = min(dis[idx5*n + idy5], dis[idx5*n + k] + dis[k*n + idy5]);
    }
    
    if (idx6 < n && idy6 < n) {
        dis[idx6*n + idy6] = min(dis[idx6*n + idy6], dis[idx6*n + k] + dis[k*n + idy6]);
    }
    
    if (idx7 < n && idy7 < n) {
        dis[idx7*n + idy7] = min(dis[idx7*n + idy7], dis[idx7*n + k] + dis[k*n + idy7]);
    }

    if (idx8 < n && idy8 < n) {
        dis[idx8*n + idy8] = min(dis[idx8*n + idy8], dis[idx8*n + k] + dis[k*n + idy8]);
    }

    if (idx9 < n && idy9 < n) {
        dis[idx9*n + idy9] = min(dis[idx9*n + idy9], dis[idx9*n + k] + dis[k*n + idy9]);
    }

    if (idx10 < n && idy10 < n) {
        dis[idx10*n + idy10] = min(dis[idx10*n + idy10], dis[idx10*n + k] + dis[k*n + idy10]);
    }

    if (idx11 < n && idy11 < n) {
        dis[idx11*n + idy11] = min(dis[idx11*n + idy11], dis[idx11*n + k] + dis[k*n + idy11]);
    }

    if (idx12 < n && idy12 < n) {
        dis[idx12*n + idy12] = min(dis[idx12*n + idy12], dis[idx12*n + k] + dis[k*n + idy12]);
    }
    
    if (idx13 < n && idy13 < n) {
        dis[idx13*n + idy13] = min(dis[idx13*n + idy13], dis[idx13*n + k] + dis[k*n + idy13]);
    }
    
    if (idx14 < n && idy14 < n) {
        dis[idx14*n + idy14] = min(dis[idx14*n + idy14], dis[idx14*n + k] + dis[k*n + idy14]);
    }
    
    if (idx15 < n && idy15 < n) {
        dis[idx15*n + idy15] = min(dis[idx15*n + idy15], dis[idx15*n + k] + dis[k*n + idy15]);
    }

}