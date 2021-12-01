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
        MxPotential pca_d = MxToCUDADevice(*p.pca);
        if(cudaMalloc(&p_d.pca, sizeof(MxPotential)) != cudaSuccess) 
            mx_error(E_FAIL, "pca malloc failed!");
        if(cudaMemcpy(p_d.pca, &pca_d, sizeof(MxPotential), cudaMemcpyHostToDevice) != cudaSuccess) 
            mx_error(E_FAIL, "pca load H2D failed!");
    }
    else 
        p_d.pca = NULL;
    if(p.pcb != NULL) { 
        MxPotential pcb_d = MxToCUDADevice(*p.pcb);
        if(cudaMalloc(&p_d.pcb, sizeof(MxPotential)) != cudaSuccess) 
            mx_error(E_FAIL, "pcb malloc failed!");
        if(cudaMemcpy(p_d.pcb, &pcb_d, sizeof(MxPotential), cudaMemcpyHostToDevice) != cudaSuccess) 
            mx_error(E_FAIL, "pcb load H2D failed!");
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
        MxPotential *pca;
        if(cudaMemcpy(pca, p->pca, sizeof(MxPotential), cudaMemcpyDeviceToHost) != cudaSuccess) 
            printf("%s\n", "pca load D2H failed!");
        Mx_cudaFree(pca);
    }
    if(p->pcb != NULL) {
        MxPotential *pcb;
        if(cudaMemcpy(pcb, p->pcb, sizeof(MxPotential), cudaMemcpyDeviceToHost) != cudaSuccess)
            printf("%s\n", "pcb load D2H failed!");
        Mx_cudaFree(pcb);
    }

    cudaFree(p->c);
    p->c = NULL;
}
