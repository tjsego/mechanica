/**
 * @file MxPotential_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines potential kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

#include "MxPotential_cuda.h"


MxPotential MxToCUDADevice(MxPotential *p) {
    MxPotential p_d(*p);

    // Alloc and copy coefficients
    if(cudaMalloc(&p_d.c, sizeof(FPTYPE) * (p->n + 1) * potential_chunk) != cudaSuccess) {
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));
        return p_d;
    }
    if(cudaMemcpy(p_d.c, p->c, sizeof(FPTYPE) * (p->n + 1) * potential_chunk, cudaMemcpyHostToDevice) != cudaSuccess) {
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));
        return p_d;
    }

    if(p->pca != NULL)
        *p_d.pca = MxToCUDADevice(p->pca);
    if(p->pcb != NULL)
        *p_d.pcb = MxToCUDADevice(p->pcb);

    return p_d;
}

