#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/BFWkernels.h"
#include "../include/graph.h"
#include "../include/core.h"
#include <iostream>


static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph) 
{
    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = BLOCK_SIZE * blockId + idy;
    const int v2 = BLOCK_SIZE * blockId + idx;

    int newPath;

    const int cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) 
    {
        cacheGraph[idy][idx] = graph[cellId];
    } 
    else 
    {
        cacheGraph[idy][idx] = INF;
    }

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) 
    {
        newPath = cacheGraph[idy][u] + cacheGraph[u][idx];

        // Synchronize before calculate new value
        __syncthreads();
        if (newPath < cacheGraph[idy][idx]) 
        {
            cacheGraph[idy][idx] = newPath;
        }

        // Synchronize to make sure that all value are current
        __syncthreads();
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = cacheGraph[idy][idx];
    }
}


static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph)
{
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = BLOCK_SIZE * blockId + idy;
    int v2 = BLOCK_SIZE * blockId + idx;

    __shared__ int cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];

    // Load base block for graph and predecessors
    int cellId = v1 * pitch + v2;

    if (v1 < nvertex && v2 < nvertex) 
    {
        cacheGraphBase[idy][idx] = graph[cellId];
    } 
    else 
    {
        cacheGraphBase[idy][idx] = INF;
    }

    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) 
    {
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    } 
    else 
    {
   // Load j-aligned singly dependent blocks
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }

    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];

    // Load current block for graph and predecessors
    int currentPath;

    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) 
    {
        currentPath = graph[cellId];
    } 
    else 
    {
        currentPath = INF;
    }
    cacheGraph[idy][idx] = currentPath;

    // Synchronize to make sure the all value are saved in cache
    __syncthreads();

    int newPath;
    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0) 
    {
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];

            if (newPath < currentPath) 
            {
                currentPath = newPath;
            }
            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    } 
    else 
    {
    // Compute j-aligned singly dependent blocks
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) 
        {
            newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];

            if (newPath < currentPath) 
            {
                currentPath = newPath;
            }

            // Synchronize to make sure that all threads compare new value with old
            __syncthreads();

           // Update new values
            cacheGraph[idy][idx] = currentPath;

           // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }

    if (v1 < nvertex && v2 < nvertex) 
    {
        graph[cellId] = currentPath;
    }
}


static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph)//, int* const pred) 
{
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;

    __shared__ int cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];

    int v1Row = BLOCK_SIZE * blockId + idy;
    int v2Col = BLOCK_SIZE * blockId + idx;

    // Load data for block
    int cellId;
    if (v1Row < nvertex && v2 < nvertex) 
    {
        cellId = v1Row * pitch + v2;

        cacheGraphBaseRow[idy][idx] = graph[cellId];
    }
    else 
    {
        cacheGraphBaseRow[idy][idx] = INF;
    }

    if (v1  < nvertex && v2Col < nvertex) 
    {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[cellId];
    }
    else 
    {
        cacheGraphBaseCol[idy][idx] = INF;
    }

    // Synchronize to make sure the all value are loaded in virtual block
   __syncthreads();

   int currentPath;
   int newPath;

   // Compute data for block
   if (v1  < nvertex && v2 < nvertex) 
   {
       cellId = v1 * pitch + v2;
       currentPath = graph[cellId];

        #pragma unroll
       for (int u = 0; u < BLOCK_SIZE; ++u) 
       {
           newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
           if (currentPath > newPath) 
           {
               currentPath = newPath;
           }
       }
       graph[cellId] = currentPath;
   }
}

void cudaBlockedFW(int nvertex, int *graphDevice, int pitch)
{

    dim3 gridPhase1(1 ,1, 1);
    dim3 gridPhase2((nvertex - 1) / BLOCK_SIZE + 1, 2 , 1);
    dim3 gridPhase3((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1 , 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

    int numBlock = (nvertex - 1) / BLOCK_SIZE + 1;

    for(int blockID = 0; blockID < numBlock; ++blockID) {
        // Start dependent phase
        _blocked_fw_dependent_ph<<<gridPhase1, dimBlockSize>>>(blockID, pitch / sizeof(int), nvertex, graphDevice);

        // Start partially dependent phase
        _blocked_fw_partial_dependent_ph<<<gridPhase2, dimBlockSize>>>(blockID, pitch / sizeof(int), nvertex, graphDevice);

        // Start independent phase
        _blocked_fw_independent_ph<<<gridPhase3, dimBlockSize>>>(blockID, pitch / sizeof(int), nvertex, graphDevice);
    }
}