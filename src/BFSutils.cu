#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void BFS_kernel(int N, int level, int *devV, int *devE, int *devD, int *devP, int *devFlag) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;
    if (thid < N && devD[thid] == level) {
        int u = thid;
        for (int i = 1; i <= devE[devV[u]]; i++) {
            int v = devE[devV[u]+i];
            if (level + 1 < devD[v]) {
                devD[v] = level + 1;
                devP[v] = i;
                valueChange = 1;
            }
        }
    }
    if (valueChange) {
        *devFlag = 1;
    }
}