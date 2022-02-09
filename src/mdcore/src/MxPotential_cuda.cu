/**
 * @file MxPotential_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines potential kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

#include "MxPotential_cuda.h"


#define cuda_call_pots_safe(func)                                       \
    if(func != cudaSuccess) {                                           \
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));    \
        return;                                                         \
    }

#define cuda_call_pots_safer(func, retval)                              \
    if(func != cudaSuccess) {                                           \
        mx_error(E_FAIL, cudaGetErrorString(cudaPeekAtLastError()));    \
        return retval;                                                  \
    }


MxPotential MxToCUDADevice(const MxPotential &p) {
    MxPotential p_d(p);

    // Alloc and copy coefficients
    cuda_call_pots_safer(cudaMalloc(&p_d.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk), p_d)
    cuda_call_pots_safer(cudaMemcpy(p_d.c, p.c, sizeof(FPTYPE) * (p.n + 1) * potential_chunk, cudaMemcpyHostToDevice), p_d)

    if(p.pca != NULL) { 
        MxPotential pca_d = MxToCUDADevice(*p.pca);
        cuda_call_pots_safer(cudaMalloc(&p_d.pca, sizeof(MxPotential)), p_d)
        cuda_call_pots_safer(cudaMemcpy(p_d.pca, &pca_d, sizeof(MxPotential), cudaMemcpyHostToDevice), p_d)
    }
    else 
        p_d.pca = NULL;
    if(p.pcb != NULL) { 
        MxPotential pcb_d = MxToCUDADevice(*p.pcb);
        cuda_call_pots_safer(cudaMalloc(&p_d.pcb, sizeof(MxPotential)), p_d)
        cuda_call_pots_safer(cudaMemcpy(p_d.pcb, &pcb_d, sizeof(MxPotential), cudaMemcpyHostToDevice), p_d)
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
    }
    if(p->pcb != NULL) {
        Mx_cudaFree(p->pcb);
    }

    cudaFree(p->c);
    p->c = NULL;
}


__host__ 
MxPotentialCUDAData::MxPotentialCUDAData(MxPotential *p) : 
    MxPotentialCUDAData()
{
    if(p == NULL) 
        return;
    
    this->kind = p->kind;
    this->flags = p->flags;
    this->alpha = make_float4(p->alpha[0], p->alpha[1], p->alpha[2], p->alpha[3]);
    this->w = make_float3(p->a, p->b, p->r0_plusone);
    this->offset = make_float3(p->offset[0], p->offset[1], p->offset[2]);
    this->n = p->n;

    if(p->kind == POTENTIAL_KIND_DPD) {
        DPDPotential* pc_dpd = (DPDPotential*)p;
        this->dpd_cfs = make_float3(pc_dpd->alpha, pc_dpd->gamma, pc_dpd->sigma);
    } 
    else {
        cuda_call_pots_safe(cudaMalloc(&this->c, sizeof(float) * (p->n + 1) * potential_chunk))
        cuda_call_pots_safe(cudaMemcpy(this->c, p->c, sizeof(float) * (p->n + 1) * potential_chunk, cudaMemcpyHostToDevice))
        this->dpd_cfs = make_float3(0.f, 0.f, 0.f);
    }
}

__host__ 
void MxPotentialCUDAData::finalize() {
    if(this->flags & POTENTIAL_NONE) 
        return;

    cuda_call_pots_safe(cudaFree(this->c))
}

__host__ 
MxPotentialCUDA::MxPotentialCUDA(MxPotential *p) : 
    MxPotentialCUDA()
{
    if(p == NULL) 
        return;

    std::vector<MxPotential*> pcs_pot, pcs;
    if(p->kind == POTENTIAL_KIND_COMBINATION) {
        for(auto pc : p->constituents()) {
            if(pc->kind != POTENTIAL_KIND_COMBINATION) {
                if(pc->kind == POTENTIAL_KIND_POTENTIAL) {
                    pcs_pot.push_back(pc);
                }
                else {
                    pcs.push_back(pc);
                }
            }
        }
    }
    else if(p->kind == POTENTIAL_KIND_POTENTIAL) {
        pcs_pot.push_back(p);
    }
    else {
        pcs.push_back(p);
    }

    this->nr_dpds = pcs.size();
    for(auto pc : pcs_pot) {
        pcs.push_back(pc);
    }

    this->nr_pots = pcs.size();

    if(this->nr_pots == 0) 
        return;
    
    MxPotentialCUDAData *data_h = (MxPotentialCUDAData*)malloc(this->nr_pots * sizeof(MxPotentialCUDAData));
    
    for(int i = 0; i < this->nr_pots; i++) {
        data_h[i] = MxPotentialCUDAData(pcs[i]);
    }

    cuda_call_pots_safe(cudaMalloc(&this->data, this->nr_pots * sizeof(MxPotentialCUDAData)))
    cuda_call_pots_safe(cudaMemcpy(this->data, data_h, this->nr_pots * sizeof(MxPotentialCUDAData), cudaMemcpyHostToDevice))

    free(data_h);
}
