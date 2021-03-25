#ifndef _CUDA_BFW_
#define _CUDA_BFW_

// CONSTS for CUDA FW
#define BLOCK_SIZE 16

void cudaNaiveFW(int nvertex, int *graph);

void cudaBlockedFW(int nvertex, int *graph);

#endif

