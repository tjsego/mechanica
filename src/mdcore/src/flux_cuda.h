/**
 * @file flux_cuda.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines flux kernels on CUDA-supporting GPUs
 * @date 2021-11-23
 * 
 */

#ifndef SRC_MDCORE_SRC_FLUX_CUDA_H_
#define SRC_MDCORE_SRC_FLUX_CUDA_H_

#include <cuda_runtime.h>

#include <mdcore_config.h>
#include <Flux.hpp>


struct MxFluxTypeIdPairCUDA {
    int16_t a;
    int16_t b;

    MxFluxTypeIdPairCUDA(TypeIdPair tip) : a{tip.a}, b{tip.b} {}
};


// A wrap of MxFlux
struct MxFluxCUDA {
    int32_t size;
    int8_t *kinds;
    MxFluxTypeIdPairCUDA *type_ids;
    int32_t *indices_a;
    int32_t *indices_b;
    float *coef;
    float *decay_coef;
    float *target;

    __host__ 
    MxFluxCUDA(MxFlux f);

    __device__ 
    void finalize();
};


// A wrap of MxFluxes
struct MxFluxesCUDA {
    int32_t size;
    MxFluxCUDA *fluxes;

    __host__ 
    MxFluxesCUDA(MxFluxes *f);

    __device__ 
    void finalize();
};


__device__ 
void MxFluxCUDA_getFluxes(unsigned int **fxind_cuda, MxFluxesCUDA **fluxes_cuda);

__device__ 
void MxFluxCUDA_getNrFluxes(unsigned int *nr_fluxes);

__device__ 
void MxFluxCUDA_getNrStates(unsigned int *nr_states);

#endif // SRC_MDCORE_SRC_FLUX_CUDA_H_