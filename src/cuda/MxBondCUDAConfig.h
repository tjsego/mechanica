/**
 * @file MxBondCUDAConfig.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for bonds
 * @date 2021-11-26
 * 
 */
#ifndef SRC_CUDA_MXBONDCUDACONFIG_H_
#define SRC_CUDA_MXBONDCUDACONFIG_H_

#include <bond_cuda.h>


/**
 * @brief CUDA runtime control interface for Mechanica bonds. 
 * 
 * This object provides control for configuring bond calculations 
 * on CUDA devices. 
 * 
 * At any time during a simulation, supported bond calculations 
 * can be sent to a particular CUDA device, or brought back to the 
 * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
 * also be specified before deploying bond calculations to a CUDA device. 
 * Future Mechanica versions will support deployment on multiple devices. 
 * 
 */
struct CAPI_EXPORT MxBondCUDAConfig {
    
    /**
     * @brief Check whether the bonds are currently on a device. 
     * 
     * @return true 
     * @return false 
     */
    static bool onDevice();

    /**
     * @brief Get the id of the device designated for running bonds. 
     * 
     * @return int 
     */
    static int getDevice();

    /**
     * @brief Set the id of the device for running bonds. 
     * 
     * Can be safely called while bonds are currently on a device. 
     * 
     * @param deviceId 
     * @return HRESULT 
     */
    static HRESULT setDevice(int deviceId=0);

    /**
     * @brief Send bonds to device. If bonds are already on device, then the call is ignored. 
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
     * Throws an error if called when the bonds are already deployed to a CUDA device. 
     * 
     * @param numBlocks number of blocks
     * @return HRESULT 
     */
    static HRESULT setBlocks(unsigned int numBlocks);
    
    /**
     * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
     * 
     * Throws an error if called when bonds are already deployed to a CUDA device. 
     * 
     * @param numThreads number of threads
     * @return HRESULT 
     */
    static HRESULT setThreads(unsigned int numThreads);

    /**
     * @brief Update a bond on a CUDA device. 
     * 
     * Useful for notifying the device that a bond has changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @param bh bond to update
     * @return HRESULT 
     */
    static HRESULT refreshBond(MxBondHandle *bh);

    /**
     * @brief Update bonds on a CUDA device. 
     * 
     * Useful for notifying the device that bonds have changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @param bonds bonds to update
     * @return HRESULT 
     */
    static HRESULT refreshBonds(std::vector<MxBondHandle*> bonds);

    /**
     * @brief Update all bonds on a CUDA device. 
     * 
     * Useful for notifying the device that bonds have changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    static HRESULT refresh();

};


#endif // SRC_CUDA_MXBONDCUDACONFIG_H_