/**
 * @file MxAngleCUDAConfig.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for angles
 * @date 2021-11-30
 * 
 */
#ifndef SRC_CUDA_MXANGLECUDACONFIG_H_
#define SRC_CUDA_MXANGLECUDACONFIG_H_

#include <angle_cuda.h>


/**
 * @brief CUDA runtime control interface for Mechanica angles. 
 * 
 * This object provides control for configuring angle calculations 
 * on CUDA devices. 
 * 
 * At any time during a simulation, supported angle calculations 
 * can be sent to a particular CUDA device, or brought back to the 
 * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
 * also be specified before deploying angle calculations to a CUDA device. 
 * Future Mechanica versions will support deployment on multiple devices. 
 * 
 */
struct CAPI_EXPORT MxAngleCUDAConfig {
    
    /**
     * @brief Check whether the angles are currently on a device. 
     * 
     * @return true 
     * @return false 
     */
    static bool onDevice();

    /**
     * @brief Get the id of the device designated for running angles. 
     * 
     * @return int 
     */
    static int getDevice();

    /**
     * @brief Send angles to device. If angles are already on device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    static HRESULT toDevice();

    /**
     * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    static HRESULT fromDevice();

    /**
     * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
     * 
     * Throws an error if called when the angles are already deployed to a CUDA device. 
     * 
     * @param numBlocks number of blocks
     * @return HRESULT 
     */
    static HRESULT setBlocks(unsigned int numBlocks);
    
    /**
     * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
     * 
     * Throws an error if called when angles are already deployed to a CUDA device. 
     * 
     * @param numThreads number of threads
     * @return HRESULT 
     */
    static HRESULT setThreads(unsigned int numThreads);

    /**
     * @brief Update a angle on a CUDA device. 
     * 
     * Useful for notifying the device that a angle has changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @param bh angle to update
     * @return HRESULT 
     */
    static HRESULT refreshAngle(MxAngleHandle *bh);

    /**
     * @brief Update angles on a CUDA device. 
     * 
     * Useful for notifying the device that angles have changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @param angles angles to update
     * @return HRESULT 
     */
    static HRESULT refreshAngles(std::vector<MxAngleHandle*> angles);

    /**
     * @brief Update all angles on a CUDA device. 
     * 
     * Useful for notifying the device that angles have changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    static HRESULT refresh();

};


#endif // SRC_CUDA_MXANGLECUDACONFIG_H_
