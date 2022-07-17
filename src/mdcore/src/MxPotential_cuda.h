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

    /** The coefficients. */
    float *c;

    __host__ 
    MxPotentialCUDAData() : flags{POTENTIAL_NONE} {}

    __host__ 
    MxPotentialCUDAData(MxPotential *p);

    __host__ 
    void finalize();
};

struct MxDPDPotentialCUDAData {
    /** Flags. */
    uint32_t flags;

    // a, b
    float2 w;

    // DPD coefficients alpha, gamma, sigma
    float3 dpd_cfs;

    __host__ 
    MxDPDPotentialCUDAData() : flags{POTENTIAL_NONE} {}

    __host__ 
    MxDPDPotentialCUDAData(DPDPotential *p);
};

// A wrap of MxPotential
struct MxPotentialCUDA {
    // Number of underlying potentials
    int nr_pots;

    // Number of dpd potentials
    int nr_dpds;

    // Data of all underlying potentials, excluding dpd
    MxPotentialCUDAData *data_pots;

    // Data of all underlying dpd potentials
    MxDPDPotentialCUDAData *data_dpds;

    __host__ __device__ 
    MxPotentialCUDA() : 
        nr_pots{0}, 
        nr_dpds{0}
    {}

    __host__ 
    MxPotentialCUDA(MxPotential *p);
    
    __host__ 
    void finalize() {
        if(this->nr_pots == 0 && this->nr_dpds) 
            return;

        for(int i = 0; i < this->nr_pots; i++) 
            this->data_pots[i].finalize();

        this->nr_pots = 0;
        this->nr_dpds = 0;
    }
};


#endif // SRC_MDCORE_SRC_MXPOTENTIAL_CUDA_H_
