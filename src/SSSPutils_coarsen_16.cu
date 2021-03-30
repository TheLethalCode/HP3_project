#include <cuda.h>
#include <cuda_runtime.h>

#define coarsen_factor 16
#define stride 32

/*

    Thread Level Coarsening 
    With Coarsening factor = 16
    And Stride Length = 32
    
    Corasening Factor will remain fixed in this file 
    Stride Length macro above can be changed for testing

*/


__global__ void SSSP_kernel1(int *V, int *E, int *W, bool *M, int *C, int *U, int n) {
    
    int tid0 = (threadIdx.x/stride)*stride*coarsen_factor  + threadIdx.x%stride + (blockIdx.x * coarsen_factor) * blockDim.x;
    int tid1 = tid0 + stride;
    int tid2 = tid1 + stride;
    int tid3 = tid2 + stride;
    int tid4 = tid3 + stride;
    int tid5 = tid4 + stride;
    int tid6 = tid5 + stride;
    int tid7 = tid6 + stride;
    int tid8 = tid7 + stride;
    int tid9 = tid8 + stride;
    int tid10 = tid9 + stride;
    int tid11 = tid10 + stride;
    int tid12 = tid11 + stride;
    int tid13 = tid12 + stride;
    int tid14 = tid13 + stride;
    int tid15 = tid14 + stride;
    
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

    if (tid4 < n && M[tid4]) {
        M[tid4] = false;
        int pos = V[tid4], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid4] + W[i]);
        }
    }

    if (tid5 < n && M[tid5]) {
        M[tid5] = false;
        int pos = V[tid5], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid5] + W[i]);
        }
    }

    if (tid6 < n && M[tid6]) {
        M[tid6] = false;
        int pos = V[tid6], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid6] + W[i]);
        }
    }

    if (tid7 < n && M[tid7]) {
        M[tid7] = false;
        int pos = V[tid7], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid7] + W[i]);
        }
    }

    if (tid8 < n && M[tid8]) {
        M[tid8] = false;
        int pos = V[tid8], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid8] + W[i]);
        }
    }

    if (tid9 < n && M[tid9]) {
        M[tid9] = false;
        int pos = V[tid9], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid9] + W[i]);
        }
    }

    if (tid10 < n && M[tid10]) {
        M[tid10] = false;
        int pos = V[tid10], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid10] + W[i]);
        }
    }

    if (tid11 < n && M[tid11]) {
        M[tid11] = false;
        int pos = V[tid11], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid11] + W[i]);
        }
    }

    if (tid12 < n && M[tid12]) {
        M[tid12] = false;
        int pos = V[tid12], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid12] + W[i]);
        }
    }

    if (tid13 < n && M[tid13]) {
        M[tid13] = false;
        int pos = V[tid13], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid13] + W[i]);
        }
    }

    if (tid14 < n && M[tid14]) {
        M[tid14] = false;
        int pos = V[tid14], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid14] + W[i]);
        }
    }

    if (tid15 < n && M[tid15]) {
        M[tid15] = false;
        int pos = V[tid15], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid15] + W[i]);
        }
    }

}

__global__ void SSSP_kernel2(bool *M, int *C, int *U, bool *flag, int n) {
    int tid0 = (threadIdx.x/stride)*stride*coarsen_factor  + threadIdx.x%stride + (blockIdx.x * coarsen_factor) * blockDim.x;
    int tid1 = tid0 + stride;
    int tid2 = tid1 + stride;
    int tid3 = tid2 + stride;
    int tid4 = tid3 + stride;
    int tid5 = tid4 + stride;
    int tid6 = tid5 + stride;
    int tid7 = tid6 + stride;
    int tid8 = tid7 + stride;
    int tid9 = tid8 + stride;
    int tid10 = tid9 + stride;
    int tid11 = tid10 + stride;
    int tid12 = tid11 + stride;
    int tid13 = tid12 + stride;
    int tid14 = tid13 + stride;
    int tid15 = tid14 + stride;

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

    if (tid4 < n) {
        if (C[tid4] > U[tid4]) {
            C[tid4] = U[tid4];
            M[tid4] = true;
            *flag = true;
        }
        U[tid4] = C[tid4];
    }

    if (tid5 < n) {
        if (C[tid5] > U[tid5]) {
            C[tid5] = U[tid5];
            M[tid5] = true;
            *flag = true;
        }
        U[tid5] = C[tid5];
    }

    if (tid6 < n) {
        if (C[tid6] > U[tid6]) {
            C[tid6] = U[tid6];
            M[tid6] = true;
            *flag = true;
        }
        U[tid6] = C[tid6];
    }

    if (tid7 < n) {
        if (C[tid7] > U[tid7]) {
            C[tid7] = U[tid7];
            M[tid7] = true;
            *flag = true;
        }
        U[tid7] = C[tid7];
    }

    if (tid8 < n) {
        if (C[tid8] > U[tid8]) {
            C[tid8] = U[tid8];
            M[tid8] = true;
            *flag = true;
        }
        U[tid8] = C[tid8];
    }

    if (tid9 < n) {
        if (C[tid9] > U[tid9]) {
            C[tid9] = U[tid9];
            M[tid9] = true;
            *flag = true;
        }
        U[tid9] = C[tid9];
    }

    if (tid10 < n) {
        if (C[tid10] > U[tid10]) {
            C[tid10] = U[tid10];
            M[tid10] = true;
            *flag = true;
        }
        U[tid10] = C[tid10];
    }

    if (tid11 < n) {
        if (C[tid11] > U[tid11]) {
            C[tid11] = U[tid11];
            M[tid11] = true;
            *flag = true;
        }
        U[tid11] = C[tid11];
    }

    if (tid12 < n) {
        if (C[tid12] > U[tid12]) {
            C[tid12] = U[tid12];
            M[tid12] = true;
            *flag = true;
        }
        U[tid12] = C[tid12];
    }

    if (tid13 < n) {
        if (C[tid13] > U[tid13]) {
            C[tid13] = U[tid13];
            M[tid13] = true;
            *flag = true;
        }
        U[tid13] = C[tid13];
    }

    if (tid14 < n) {
        if (C[tid14] > U[tid14]) {
            C[tid14] = U[tid14];
            M[tid14] = true;
            *flag = true;
        }
        U[tid14] = C[tid14];
    }

    if (tid15 < n) {
        if (C[tid15] > U[tid15]) {
            C[tid15] = U[tid15];
            M[tid15] = true;
            *flag = true;
        }
        U[tid15] = C[tid15];
    }
}