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
    // v[0], v[1], v[2], radius
    float4 v;

    // id, typeId, clusterId, flags
    int4 w;

    __host__ __device__ 
    MxParticleCUDA()  :
        w{-1, -1, -1, PARTICLE_NONE}
    {}

    __host__ __device__ 
    MxParticleCUDA(MxParticle *p) : 
        x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
        v{p->v[0], p->v[1], p->v[2], p->radius}, 
        w{p->id, p->typeId, p->clusterId, p->flags}
    {}

    __host__ 
    MxParticleCUDA(MxParticle *p, int nr_states) : 
        x{p->x[0], p->x[1], p->x[2], p->x[3]}, 
        v{p->v[0], p->v[1], p->v[2], p->radius}, 
        w{p->id, p->typeId, p->clusterId, p->flags}
    {}
};

#endif // SRC_MDCORE_SRC_MXPARTICLE_CUDA_H_
