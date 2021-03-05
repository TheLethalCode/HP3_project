#include <cuda.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <cstdio>

#include "graph.h"
#include "bfsCPU.h"

__global__ void cudaBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
               int *d_edgesSize, int *d_distance, int *d_parent, int *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if (thid < N && d_distance[thid] == level) {
        int u = thid;
        for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}


void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
    std::vector<int> &parent, std::vector<bool> &visited) {
printf("Starting sequential bfs.\n");
auto start = std::chrono::steady_clock::now();
bfsCPU(startVertex, G, distance, parent, visited);
auto end = std::chrono::steady_clock::now();
long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

void checkError(cudaError_t error, std::string msg) {
    if (error != CUDA_SUCCESS) {
        printf("%s: %d\n", msg.c_str(), error);
        exit(1);
    }
}

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

int *d_adjacencyList;
int *d_edgesOffset;
int *d_edgesSize;
int *d_distance;
int *d_parent;

void initCuda(Graph &G) {
    //copy memory to device
    checkError(cudaMalloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
    checkError(cudaMalloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
    checkError(cudaMalloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
    checkError(cudaMalloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
    checkError(cudaMalloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
    checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_adjacencyList");
    checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_edgesOffset");
    checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_edgesSize");

}

void finalizeCuda() {
    // free memory
    checkError(cudaFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
    checkError(cudaFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
    checkError(cudaFree(d_edgesSize), "cannot free memory for d_edgesSize");
    checkError(cudaFree(d_distance), "cannot free memory for d_distance");
    checkError(cudaFree(d_parent), "cannot free memory for d_parent");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //initialize values
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    checkError(cudaMemcpy(d_distance, distance.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d)distance");
    checkError(cudaMemcpy(d_parent, parent.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice),
               "cannot copy to d_parent");
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Graph &G) {
    //copy memory from device
    checkError(cudaMemcpy(distance.data(), d_distance, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost),
               "cannot copy d_distance to host");
    checkError(cudaMemcpy(parent.data(), d_parent, G.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cannot copy d_parent to host");

}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<int> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *changed;
    checkError(cudaMalloc((void **) &changed, sizeof(int)), "cannot allocate changed");

    //launch kernel
    printf("Starting simple parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;
        int threadsPerBlock = 1024;
        int blocksPerGrid = (G.numVertices)/ threadsPerBlock + 1;
        cudaBfs<<<blocksPerGrid, threadsPerBlock>>>(G.numVertices, level, d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance, d_parent, changed);
        cudaDeviceSynchronize();
        level++;
    }


    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}


int main(int argc, char **argv){
    Graph G;
    readGraph(G, argc, argv);

    int startVertex = atoi(argv[1]);

    printf("Number of vertices %d\n", G.numVertices);
    printf("Number of edges %d\n\n", G.numEdges);
    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    //run CPU sequential bfs
    runCpu(startVertex, G, distance, parent, visited);

    initCuda(G);
    //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);

    finalizeCuda();
}
