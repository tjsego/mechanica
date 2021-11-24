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

__device__ 
void flux_eval_ex_cuda(unsigned int typeTableIndex, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i, bool *result);

__device__ 
void flux_eval_ex_cuda(unsigned int typeTableIndex, float r, float *states_i, float *states_j, int type_i, int type_j, float *qvec_i, float *qvec_j, bool *result);

__device__ 
void MxFluxCUDA_getPartStates(float **result);

__device__ 
void MxFluxCUDA_getNrFluxes(unsigned int *nr_fluxes);

__device__ 
void MxFluxCUDA_getNrStates(unsigned int *nr_states);

__global__ 
void MxFluxCUDA_copy_partstates(float *states, int count, int ind, int nr_states);

#endif // SRC_MDCORE_SRC_FLUX_CUDA_H_