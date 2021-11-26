/**
 * @file MxParticle_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines particle kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

#ifndef SRC_MDCORE_SRC_MXPARTICLE_CUDA_H_
#define SRC_MDCORE_SRC_MXPARTICLE_CUDA_H_

#include "MxParticle.h"

#include <cuda_runtime.h>


// A wrap of MxParticle
struct MxParticleCUDA {
    float4 x;
    float4 v;
    float radius;
    int id;
    int typeId;
    int clusterId;
    uint16_t flags;
    float states[MX_SIMD_SIZE];

    __host__ __device__ 
    MxParticleCUDA()  :
        id{-1}, 
        typeId{-1}
    {}

    __host__ __device__ 
    MxParticleCUDA(MxParticle *p) : 
        x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
        v{p->v[0], p->v[1], p->v[2], p->v[3]}, 
        radius{p->radius}, 
        id{p->id}, 
        typeId{p->typeId}, 
        clusterId{p->clusterId}, 
        flags{p->flags}
    {}

    __host__ 
    MxParticleCUDA(MxParticle *p, int nr_states) : 
        x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
        v{p->v[0], p->v[1], p->v[2], p->v[3]}, 
        radius{p->radius}, 
        id{p->id}, 
        typeId{p->typeId}, 
        clusterId{p->clusterId}, 
        flags{p->flags}
    {
        memcpy(this->states, p->state_vector->fvec, sizeof(float) * nr_states);
    }
};

#endif // SRC_MDCORE_SRC_MXPARTICLE_CUDA_H_
