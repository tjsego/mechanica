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
    
    this->flags = p->flags;
    this->alpha = make_float4(p->alpha[0], p->alpha[1], p->alpha[2], p->alpha[3]);
    this->w = make_float3(p->a, p->b, p->r0_plusone);
    this->offset = make_float3(p->offset[0], p->offset[1], p->offset[2]);
    this->n = p->n;

    cuda_call_pots_safe(cudaMalloc(&this->c, sizeof(float) * (p->n + 1) * potential_chunk))
    cuda_call_pots_safe(cudaMemcpy(this->c, p->c, sizeof(float) * (p->n + 1) * potential_chunk, cudaMemcpyHostToDevice))
}

__host__ 
void MxPotentialCUDAData::finalize() {
    if(this->flags & POTENTIAL_NONE) 
        return;

    cuda_call_pots_safe(cudaFree(this->c))
}

__host__ 
MxDPDPotentialCUDAData::MxDPDPotentialCUDAData(DPDPotential *p) : 
    MxDPDPotentialCUDAData()
{
    if(p == NULL) 
        return;

    this->flags = p->flags;
    this->w = make_float2(p->a, p->b);
    this->dpd_cfs = make_float3(p->alpha, p->gamma, p->sigma);
}

__host__ 
MxPotentialCUDA::MxPotentialCUDA(MxPotential *p) : 
    MxPotentialCUDA()
{
    if(p == NULL) 
        return;

    std::vector<MxPotential*> pcs_pot;
    std::vector<DPDPotential*> pcs_dpd;
    if(p->kind == POTENTIAL_KIND_COMBINATION) {
        for(auto pc : p->constituents()) {
            if(pc->kind != POTENTIAL_KIND_COMBINATION) {
                if(pc->kind == POTENTIAL_KIND_POTENTIAL) {
                    pcs_pot.push_back(pc);
                }
                else {
                    pcs_dpd.push_back((DPDPotential*)pc);
                }
            }
        }
    }
    else if(p->kind == POTENTIAL_KIND_POTENTIAL) {
        pcs_pot.push_back(p);
    }
    else {
        pcs_dpd.push_back((DPDPotential*)p);
    }

    this->nr_dpds = pcs_dpd.size();
    this->nr_pots = pcs_pot.size();

    if(this->nr_pots == 0 && this->nr_dpds == 0) 
        return;
    
    MxPotentialCUDAData *data_h_pots = (MxPotentialCUDAData*)malloc(this->nr_pots * sizeof(MxPotentialCUDAData));
    MxDPDPotentialCUDAData *data_h_dpds = (MxDPDPotentialCUDAData*)malloc(this->nr_dpds * sizeof(MxDPDPotentialCUDAData));
    
    for(int i = 0; i < this->nr_pots; i++) {
        data_h_pots[i] = MxPotentialCUDAData(pcs_pot[i]);
    }
    for(int i = 0; i < this->nr_dpds; i++) {
        data_h_dpds[i] = MxDPDPotentialCUDAData(pcs_dpd[i]);
    }

    cuda_call_pots_safe(cudaMalloc(&this->data_pots, this->nr_pots * sizeof(MxPotentialCUDAData)))
    cuda_call_pots_safe(cudaMemcpy(this->data_pots, data_h_pots, this->nr_pots * sizeof(MxPotentialCUDAData), cudaMemcpyHostToDevice))

    cuda_call_pots_safe(cudaMalloc(&this->data_dpds, this->nr_dpds * sizeof(MxDPDPotentialCUDAData)))
    cuda_call_pots_safe(cudaMemcpy(this->data_dpds, data_h_dpds, this->nr_dpds * sizeof(MxDPDPotentialCUDAData), cudaMemcpyHostToDevice))

    free(data_h_pots);
    free(data_h_dpds);
}
