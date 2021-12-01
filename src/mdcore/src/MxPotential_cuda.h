/**
 * @file MxPotential_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines potential kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

#ifndef SRC_MDCORE_SRC_MXPOTENTIAL_CUDA_H_
#define SRC_MDCORE_SRC_MXPOTENTIAL_CUDA_H_

#include "MxPotential.h"
#include "DissapativeParticleDynamics.hpp"

#include "MxParticle_cuda.h"


/**
 * @brief Loads a potential onto a CUDA device
 * 
 * @param p The potential
 * 
 * @return The loaded potential, or NULL if failed
 */
MxPotential MxToCUDADevice(const MxPotential &p);


__host__ __device__ 
void Mx_cudaFree(MxPotential *p);


// A wrap of MxPotential
struct MxPotentialCUDA {

    // Flag signifying whether instance is a placeholder
    bool empty;

    // DPD coefficients alpha, gamma, sigma
    float3 dpd_cfs;

    // The potential
    MxPotential pot;

    __host__ __device__ 
    MxPotentialCUDA() :
        empty{true}
    {}

    __host__ __device__ 
    MxPotentialCUDA(const MxPotential &p, bool toDevice=true) {
        this->empty = false;

        #if defined(__CUDA_ARCH__)
        this->pot = p;
        #else
        if(toDevice) this->pot = MxToCUDADevice(p);
        else this->pot = p;
        #endif

        if(p.kind == POTENTIAL_KIND_DPD) {
            DPDPotential *p_dpd = (DPDPotential*)&p;
            this->dpd_cfs.x = p_dpd->alpha;
            this->dpd_cfs.y = p_dpd->gamma;
            this->dpd_cfs.z = p_dpd->sigma;
        }
    }
    
    __host__ 
    void finalize() {
        if(!this->empty) Mx_cudaFree(&this->pot);
        this->empty = true;
    }
};


#endif // SRC_MDCORE_SRC_MXPOTENTIAL_CUDA_H_
