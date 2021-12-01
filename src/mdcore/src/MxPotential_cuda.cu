/**
 * @file MxPotential_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines potential kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

#include "MxPotential_cuda.h"


MxPotential MxToCUDADevice(const MxPotential &p) {
    MxPotential p_d(p);

    // Alloc and copy coefficients
    if(cudaMalloc(&p_d.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk) != cudaSuccess) {
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));
        return p_d;
    }
    if(cudaMemcpy(p_d.c, p.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk, cudaMemcpyHostToDevice) != cudaSuccess) {
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));
        return p_d;
    }

    if(p.pca != NULL) { 
        auto pca_d = new MxPotential(MxToCUDADevice(*p.pca));
        p_d.pca = pca_d;
    }
    else 
        p_d.pca = NULL;
    if(p.pcb != NULL) { 
        auto pcb_d = new MxPotential(MxToCUDADevice(*p.pcb));
        p_d.pcb = pcb_d;
    } 
    else 
        p_d.pcb = NULL;

    return p_d;
}

__host__ __device__ 
void Mx_cudaFree(MxPotential *p) {
    if(p == NULL || p->flags & POTENTIAL_NONE) 
        return;
    
    if(p->pca != NULL) {
        Mx_cudaFree(p->pca);
        delete p->pca;
    }
    if(p->pcb != NULL) {
        Mx_cudaFree(p->pcb);
        delete p->pcb;
    }

    cudaFree(p->c);
    p->c = NULL;
}
