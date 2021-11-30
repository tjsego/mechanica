/**
 * @file MxSimulatorCUDAConfig.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for simulator
 * @date 2021-11-10
 * 
 */
#ifndef SRC_CUDA_MXSIMULATORCUDACONFIG_H_
#define SRC_CUDA_MXSIMULATORCUDACONFIG_H_

#include <mx_cuda.h>

#include "MxEngineCUDAConfig.h"
#include "MxBondCUDAConfig.h"
#include "MxAngleCUDAConfig.h"


/**
 * @brief CUDA runtime control interface for MxSimulator. 
 * 
 * This object aggregates all CUDA runtime control interfaces relevant 
 * to a Mechanica simulation. 
 * 
 */
struct CAPI_EXPORT MxSimulatorCUDAConfig {
    /** Mechanica engine CUDA runtime control interface */
    MxEngineCUDAConfig engine;

    /** Mechanica bonds CUDA runtime control interface */
    MxBondCUDAConfig bonds;

    /** Mechanica angles CUDA runtime control interface */
    MxAngleCUDAConfig angles;
};

#endif // SRC_CUDA_MXSIMULATORCUDACONFIG_H_
