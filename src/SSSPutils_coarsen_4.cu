#include <cuda.h>
#include <cuda_runtime.h>

#define coarsen_factor 4
#define stride 32

/*

    Thread Level Coarsening 
    With Coarsening factor of 4 
    And Stride Length of 32
    
    Corasening Factor will reamin fixed in this file 
    Stride Length macro above can be changed for testing

*/


__global__ void SSSP_kernel1(int *V, int *E, int *W, bool *M, int *C, int *U, int n) {
    
    int tid0 = (threadIdx.x/stride)*stride*coarsen_factor  + threadIdx.x%stride + (blockIdx.x * coarsen_factor) * blockDim.x;
    
    int tid1 = tid0 + stride;
    int tid2 = tid1 + stride;
    int tid3 = tid2 + stride;
    
    if (tid0 < n && M[tid0]) {
        M[tid0] = false;
        int pos = V[tid0], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid0] + W[i]);
        }
    }

    if (tid1 < n && M[tid1]) {
        M[tid1] = false;
        int pos = V[tid1], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid1] + W[i]);
        }
    }

    if (tid2 < n && M[tid2]) {
        M[tid2] = false;
        int pos = V[tid2], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid2] + W[i]);
        }
    }

    if (tid3 < n && M[tid3]) {
        M[tid3] = false;
        int pos = V[tid3], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid3] + W[i]);
        }
    }
}

__global__ void SSSP_kernel2(bool *M, int *C, int *U, bool *flag, int n) {
    int tid0 = (threadIdx.x/stride)*stride*coarsen_factor  + threadIdx.x%stride + (blockIdx.x * coarsen_factor) * blockDim.x;
    int tid1 = tid0 + stride;
    int tid2 = tid1 + stride;
    int tid3 = tid2 + stride;

    if (tid0 < n) {
        if (C[tid0] > U[tid0]) {
            C[tid0] = U[tid0];
            M[tid0] = true;
            *flag = true;
        }
        U[tid0] = C[tid0];
    }

    if (tid1 < n) {
        if (C[tid1] > U[tid1]) {
            C[tid1] = U[tid1];
            M[tid1] = true;
            *flag = true;
        }
        U[tid1] = C[tid1];
    }

    if (tid2 < n) {
        if (C[tid2] > U[tid2]) {
            C[tid2] = U[tid2];
            M[tid2] = true;
            *flag = true;
        }
        U[tid2] = C[tid2];
    }

    if (tid3 < n) {
        if (C[tid3] > U[tid3]) {
            C[tid3] = U[tid3];
            M[tid3] = true;
            *flag = true;
        }
        U[tid3] = C[tid3];
    }
}