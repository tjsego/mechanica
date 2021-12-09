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


struct MxPotentialCUDAData {
    uint32_t kind;

    /** Flags. */
    uint32_t flags;

    // a, b, r0_plusone
    float3 w;

    /** coordinate offset */
    float3 offset;

    /** Coefficients for the interval transform. */
    float4 alpha;

    /** Nr of intervals. */
    int n;

    // DPD coefficients alpha, gamma, sigma
    float3 dpd_cfs;

    /** The coefficients. */
    float *c;

    __host__ 
    MxPotentialCUDAData() : flags{POTENTIAL_NONE} {}

    __host__ 
    MxPotentialCUDAData(MxPotential *p);

    __host__ 
    void finalize();
};

// A wrap of MxPotential
struct MxPotentialCUDA {
    // Number of underlying potentials
    int nr_pots;

    // Number of dpd potentials
    int nr_dpds;

    // Data of all underlying potentials
    MxPotentialCUDAData *data;

    __host__ __device__ 
    MxPotentialCUDA() : 
        nr_pots{0}, 
        nr_dpds{0}
    {}

    __host__ 
    MxPotentialCUDA(MxPotential *p);
    
    __host__ 
    void finalize() {
        if(this->nr_pots == 0) 
            return;

        for(int i = 0; i < this->nr_pots; i++) 
            this->data[i].finalize();

        this->nr_pots = 0;
    }
};


#endif // SRC_MDCORE_SRC_MXPOTENTIAL_CUDA_H_
