#include "core.h"
#include "graph.h"
#include "shortestPathCPU.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <utility>
#include <string>

#define NUM_THREADS 256

__global__ void SSSP_kernel1(int *V, int *E, int *W, bool *M, int *C, int *U, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && M[tid]) {
        M[tid] = false;
        int pos = V[tid], size = E[pos];
        for (int i = pos + 1; i < pos + size + 1; i++) {
            int nid = E[i];
            atomicMin(&U[nid], C[tid] + W[i]);
        }
    }
}

__global__ void SSSP_kernel2(bool *M, int *C, int *U, bool *flag, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        if (C[tid] > U[tid]) {
            C[tid] = U[tid];
            M[tid] = true;
            *flag = true;
        }
        U[tid] = C[tid];
    }
}

template<typename T>
void allocCopy(T **devV, T *V, int n, std::string s) {
    if (cudaCheck(cudaMalloc((void **)devV, sizeof(T)*n)) &&
          cudaCheck(cudaMemcpy(*devV, V, sizeof(T)*n, cudaMemcpyHostToDevice))) {
        std::cout << "Allocated memory and copied " << s << " to device" << std::endl;
    }
}

int main(int argc, char* argv[]) {

    // Take Input or Generate Graph
    Graph G;
    int s;
    if (argc == 1 || atoi(argv[1]) != 1) {
        int n, m;
        std::cin >> n >> m;
        G.readGraph(n, m);
    } else if (argc >= 4) {
        int n = atoi(argv[2]), m = atoi(argv[3]), lim, seed;
        lim = (argc >= 5? atoi(argv[4]): 20);
        seed = (argc >= 6? atoi(argv[5]): 81);
        G.genGraph(n, m, lim, seed);
    } else {
        std::cerr << "Incorrect arguments " << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Source Vertex: ";
    std::cin >> s;

      // ========================= CUDA ============================= //
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Declare and Initialise Host Array
    int *V, *E, *W, *C, *U, Vs, Es;
    bool *M, flag;
    Vs = vecToArr(G.posV, &V);
    Es = vecToArr(G.packE, &E);
    vecToArr(G.packW, &W);
    C = new int[Vs];
    U = new int[Vs];
    M = new bool[Vs];
    std::fill_n(C, Vs, INF);
    std::fill_n(U, Vs, INF);
    std::fill_n(M, Vs, false);

    // Update source values
    C[s] = U[s] = 0;
    M[s] = flag = true;

    // Declare and Initialise Device Array
    int *devV, *devE, *devW, *devC, *devU;
    bool *devM, *devFlag; 
    allocCopy<int>(&devV, V, Vs, "V_a");
    allocCopy<int>(&devE, E, Es, "E_a");
    allocCopy<int>(&devW, W, Es, "W_a");
    allocCopy<int>(&devC, C, Vs, "C_a");
    allocCopy<int>(&devU, U, Vs, "U_a");
    allocCopy<bool>(&devM, M, Vs, "M_a");
    allocCopy<bool>(&devFlag, &flag, 1, "flag");
    
    // Run Cuda Parallel
    int blocks = (Vs + NUM_THREADS - 1) / NUM_THREADS;
    cudaEventRecord(start);
    while (flag) {
        flag = false;
        cudaMemcpy(devFlag, &flag, sizeof(bool), cudaMemcpyHostToDevice);
        SSSP_kernel1<<< blocks, NUM_THREADS >>>(devV, devE, devW, devM, devC, devU, Vs);
        SSSP_kernel2<<< blocks, NUM_THREADS >>>(devM, devC, devU, devFlag, Vs);
        cudaMemcpy(&flag, devFlag, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaCheck(cudaPeekAtLastError());
    if (cudaCheck(cudaMemcpy(C, devC, Vs * sizeof(int), cudaMemcpyDeviceToHost))) {
        std::cout << "Obtained distance in host at C_a" << std::endl;
    }

    // Calculate Time Taken
    cudaEventSynchronize(stop);
    float timeGPU = 0;
    cudaEventElapsedTime(&timeGPU, start, stop);
    std::cout << "CUDA Elapsed Time (in ms): " << timeGPU << std::endl;

    // ========================= CPU ============================= //
    int *dis = new int[Vs];
    auto beg = std::chrono::high_resolution_clock::now();
    djikstra(G, s, dis);
    auto end = std::chrono::high_resolution_clock::now();
    float timeCPU = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
    std::cout << "CPU Elapsed Time (in ms): " << timeCPU / 1000 << std::endl;

    // ======================= Verification ==========================//
    for (int i = 0; i < Vs; i++) {
        if (dis[i] != C[i]) {
            std::cout << "Not a Match at " << i << std::endl;
            std::cout << "GPU dist: " << C[i] << std::endl;
            std::cout << "CPU dist: " << dis[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}