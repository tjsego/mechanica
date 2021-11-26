/**
 * @file MxBoundaryConditions_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines boundary condition kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */
#ifndef SRC_MDCORE_SRC_MXBOUNDARYCONDITIONS_CUDA_H_
#define SRC_MDCORE_SRC_MXBOUNDARYCONDITIONS_CUDA_H_

#include "MxBoundaryConditions.hpp"

#include "MxPotential_cuda.h"


struct MxBoundaryConditionCUDA {

    float3 normal;

    float3 velocity;

    float radius;

    MxPotentialCUDA *pots;

    __host__ __device__ 
    MxBoundaryConditionCUDA() {}

    __host__ 
    MxBoundaryConditionCUDA(const MxBoundaryCondition &_bc);

    __device__ 
    void finalize();
};


struct MxBoundaryConditionsCUDA {
    
    // Left, right, front, back, bottom, top
    MxBoundaryConditionCUDA *bcs;

    __host__ __device__ 
    MxBoundaryConditionsCUDA() {}
    
    __host__ 
    MxBoundaryConditionsCUDA(const MxBoundaryConditions &_bcs);

    __device__ 
    void finalize();

};

#endif // SRC_MDCORE_SRC_MXBOUNDARYCONDITIONS_CUDA_H_
