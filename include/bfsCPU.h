#ifndef BFS_CUDA_BFSCPU_H
#define BFS_CUDA_BFSCPU_H

#include <chrono>
#include <queue>
#include "graph.h"

void bfsCPU(int start, Graph &G, int *distance,
            int *parent, bool *visited);

#endif //BFS_CUDA_BFSCPU_H