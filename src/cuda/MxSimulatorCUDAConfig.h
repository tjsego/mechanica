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
};

#endif // SRC_CUDA_MXSIMULATORCUDACONFIG_H_
