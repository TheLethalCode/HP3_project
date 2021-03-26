#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#define NUM_THREADS 16
// ========================= Parallel BFS ============================= //

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

// ========================= Queue BFS ============================ //

__global__ void queueBfs(int level, int *devV, int *devE, int *devD, int *devP,
              int queueSize, int *nextQueueSize, int *devCurrentQueue, int *devNextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = devCurrentQueue[thid];
        for (int i = 1; i <= devE[devV[u]]; i++) {
            int v = devE[devV[u]+i];
            if (devD[v] == INT_MAX && atomicMin(&devD[v], level + 1) == INT_MAX) {
                devP[v] = devV[u]+i;
                int position = atomicAdd(nextQueueSize, 1);
                devNextQueue[position] = v;
            }
        }
    }
}


// ========================= Scan BFS ============================= //

__global__ void nextLayer(int level, int *devV, int *devE, int *devP, int *devD, int queueSize, int *devCurrentQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < queueSize) {
        int u = devCurrentQueue[thid];
        for (int i = 1; i <= devE[devV[u]]; i++) {
            int pos = devV[u]+i;
            int v = devE[pos];
            if (level + 1 < devD[v]) {
                devD[v] = level + 1;
                devP[v] = pos;
            }
        }
    }

}

__global__ void countDegrees(int *devV, int *devE, int *devP, int queueSize, int *devCurrentQueue, int *devDegrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < queueSize) {
        int u = devCurrentQueue[thid];
        int degree = 0;
        for (int i = 1; i <= devE[devV[u]]; i++) {
            int pos = devV[u]+i;
            int v = devE[pos];
            if (devP[v] == (pos) && v != u) {
                ++degree;
            }
        }
        devDegrees[thid] = degree;
    }
}

__global__ void scanDegrees(int N, int *devDegrees, int *incrDegrees, int *num) {
    //TODO: copied this part, need to understand
    int thid = blockIdx.x * blockDim.x + threadIdx.x; 
    printf("scanDegrees thid %d\n", thid);
    printf("scanDegrees thid %d\n", thid);
    // unsigned int fakeN = N;
    int val = atomicAdd(num, 1);
    printf("\n [ fakeN %d val %d ] \n", *num, val);
    if (thid < NUM_THREADS) {
        //write initial values to shared memory
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = devDegrees[thid];
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < N) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
        }

        //go down
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < N) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        devDegrees[thid] = prefixSum[modulo];
    }
}

__global__ void assignVerticesNextQueue(int *devV, int *devE, int *devP, int queueSize,
                             int *devCurrentQueue, int *devNextQueue, int *devDegrees, int *incrDegrees,
                             int nextQueueSize) {
    //TODO: copied this part, need to understand
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("assignVerticesNextQ thid %d\n", thid);
    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = devDegrees[thid - 1];
        }

        int u = devCurrentQueue[thid];
        int counter = 0;
        for (int i = 1; i <= devE[devV[u]]; i++) {
            int v = devE[devV[u]+i];
            if (devP[v] == devV[u]+i && v != u) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                // printf("nextQplace %d u %d v %d counter %d\n", nextQueuePlace, u, v, counter);
                devNextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}