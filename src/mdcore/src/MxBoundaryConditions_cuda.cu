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

    this->pots_h = (MxPotentialCUDA*)malloc(size_pots);
    for(int typeId = 0; typeId < engine_maxnrtypes; typeId++) { 
        p = _bc.potenntials[typeId];
        if(p != NULL) 
            this->pots_h[typeId] = MxPotentialCUDA(*p);
        else 
            this->pots_h[typeId] = MxPotentialCUDA();
    }

    if(cudaMemcpy(this->pots, this->pots_h, size_pots, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("Boundary condition copy H2D failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}

__host__ 
void MxBoundaryConditionCUDA::finalize() {
    for(int typeId = 0; typeId < engine_maxnrtypes; typeId++) {
        auto p = this->pots_h[typeId];
        p.finalize();
    }

    free(this->pots_h);
    
    if(cudaFree(this->pots) != cudaSuccess) 
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

    this->bcs_h = (MxBoundaryConditionCUDA*)malloc(size_bcs);

    this->bcs_h[0] = MxBoundaryConditionCUDA(_bcs.left);
    this->bcs_h[1] = MxBoundaryConditionCUDA(_bcs.right);
    this->bcs_h[2] = MxBoundaryConditionCUDA(_bcs.front);
    this->bcs_h[3] = MxBoundaryConditionCUDA(_bcs.back);
    this->bcs_h[4] = MxBoundaryConditionCUDA(_bcs.bottom);
    this->bcs_h[5] = MxBoundaryConditionCUDA(_bcs.top);

    if(cudaMemcpy(this->bcs, this->bcs_h, size_bcs, cudaMemcpyHostToDevice) != cudaSuccess)
        printf("Boundary conditions copy H2D failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}

__host__ 
void MxBoundaryConditionsCUDA::finalize() {
    for(int bcId = 0; bcId < 6; bcId++)
        this->bcs_h[bcId].finalize();

    free(this->bcs_h);

    if(cudaFree(this->bcs) != cudaSuccess) 
        printf("Boundary conditions finalize failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
}
