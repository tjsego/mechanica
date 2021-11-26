/**
 * @file MxBoundaryConditions_cuda.cu
 * @author T.J. Sego, Ph.D.
 * @brief Defines boundary condition kernels on CUDA-supporting GPUs
 * @date 2021-11-24
 * 
 */

// TODO: improve error handling in MxBoundaryConditions_cuda

#include "MxBoundaryConditions_cuda.h"

#include "engine.h"


// MxBoundaryConditionCUDA


__host__ 
MxBoundaryConditionCUDA::MxBoundaryConditionCUDA(const MxBoundaryCondition &_bc) {
    this->normal = make_float3(_bc.normal[0], _bc.normal[1], _bc.normal[2]);
    this->velocity = make_float3(_bc.velocity[0], _bc.velocity[1], _bc.velocity[2]);
    this->radius = _bc.radius;

    MxPotential *p;
    
    size_t size_pots = sizeof(MxPotentialCUDA) * engine_maxnrtypes;
    if(cudaMalloc(&this->pots, size_pots) != cudaSuccess) {
        printf("Boundary condition allocation failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        return;
    }

    MxPotentialCUDA *cu_pots = (MxPotentialCUDA*)malloc(size_pots);
    for(int typeId = 0; typeId < engine_maxnrtypes; typeId++) 
        if((p = _bc.potenntials[typeId]) != NULL) 
            cu_pots[typeId] = MxPotentialCUDA(p);

    if(cudaMemcpy(this->pots, cu_pots, size_pots, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("Boundary condition copy H2D failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    
    free(cu_pots);
}

__device__ 
void MxBoundaryConditionCUDA::finalize() {
    for(int typeId = 0; typeId < engine_maxnrtypes; typeId++)
        this->pots[typeId].finalize();
    
    if(cudaFree(&this->pots) != cudaSuccess) 
        printf("Boundary condition finalize failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}


// MxBoundaryConditionsCUDA


__host__ 
MxBoundaryConditionsCUDA::MxBoundaryConditionsCUDA(const MxBoundaryConditions &_bcs) {
    size_t size_bcs = sizeof(MxBoundaryConditionCUDA) * 6;

    if(cudaMalloc(&this->bcs, size_bcs) != cudaSuccess) {
        printf("Boundary conditions allocation failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        return;
    }

    MxBoundaryConditionCUDA *cu_bcs = (MxBoundaryConditionCUDA*)malloc(size_bcs);

    cu_bcs[0] = MxBoundaryConditionCUDA(_bcs.left);
    cu_bcs[1] = MxBoundaryConditionCUDA(_bcs.right);
    cu_bcs[2] = MxBoundaryConditionCUDA(_bcs.front);
    cu_bcs[3] = MxBoundaryConditionCUDA(_bcs.back);
    cu_bcs[4] = MxBoundaryConditionCUDA(_bcs.bottom);
    cu_bcs[5] = MxBoundaryConditionCUDA(_bcs.top);

    if(cudaMemcpy(this->bcs, cu_bcs, size_bcs, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("Boundary conditions copy H2D failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

    free(cu_bcs);
}

__device__ 
void MxBoundaryConditionsCUDA::finalize() {
    for(int bcId = 0; bcId < 6; bcId++)
        this->bcs[bcId].finalize();

    if(cudaFree(&this->bcs) != cudaSuccess) 
        printf("Boundary conditions finalize failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}
