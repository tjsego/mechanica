/**
 * @file MxEngineCUDAConfig.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines CUDA runtime control interface for engine
 * @date 2021-11-10
 * 
 */
#ifndef SRC_CUDA_MXENGINECUDACONFIG_H_
#define SRC_CUDA_MXENGINECUDACONFIG_H_

#include <mx_cuda.h>


/**
 * @brief CUDA runtime control interface for Mechanica engine. 
 * 
 * This object provides control for configuring engine calculations 
 * on CUDA devices. Associated calculations include nonbonded particle 
 * interactions, sorting and space partitioning. 
 * 
 * At any time during a simulation, supported engine calculations 
 * can be sent to a particular CUDA device, or brought back to the 
 * CPU when deployed on a CUDA device. CUDA dynamic parallelism can 
 * also be specified before deploying engine calculations to a CUDA device. 
 * Future Mechanica versions will support deployment on multiple devices. 
 * 
 */
struct CAPI_EXPORT MxEngineCUDAConfig {
    
    MxEngineCUDAConfig();
    ~MxEngineCUDAConfig() {}

    /**
     * @brief Check whether the engine is currently on a device. 
     * 
     * @return true 
     * @return false 
     */
    bool onDevice();

    /**
     * @brief Get the id of the device running the engine. 
     * 
     * Returns -1 if engine is not on a device. 
     * 
     * @return int 
     */
    int getDevice();

    /**
     * @brief Set the id of the device for running the engine. 
     * 
     * Fails if engine is currently on a device. 
     * 
     * @param deviceId 
     * @return HRESULT 
     */
    HRESULT setDevice(int deviceId=0);

    /**
     * @brief Clear configured device for the engine. 
     * 
     * Fails if engine is currently on a device. 
     * 
     * @return HRESULT 
     */
    HRESULT clearDevice();

    /**
     * @brief Send engine to device. If engine is already on device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    HRESULT toDevice();

    /**
     * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    HRESULT fromDevice();

    /**
     * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
     * 
     * Throws an error if called when the engine is already deployed to a CUDA device. 
     * 
     * @param numBlocks number of blocks
     * @param deviceId device ID (optional)
     * @return HRESULT 
     */
    HRESULT setBlocks(unsigned int numBlocks, int deviceId=-1);
    
    /**
     * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
     * 
     * Throws an error if called when the engine is already deployed to a CUDA device. 
     * 
     * @param numThreads number of threads
     * @param deviceId device ID (optional)
     * @return HRESULT 
     */
    HRESULT setThreads(unsigned int numThreads, int deviceId=-1);

    /**
     * @brief Update potentials on a CUDA device. 
     * 
     * Useful for notifying the device that a potential has changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    HRESULT refreshPotentials();

    /**
     * @brief Update boundary conditions on a CUDA device. 
     * 
     * Useful for notifying the device that a boundary condition has changed. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    HRESULT refreshBoundaryConditions();

    /**
     * @brief Update the image of the engine on a CUDA device. 
     * 
     * Necessary to notify the device of changes to engine data that 
     * are not automatically handled by Mechanica. Refer to documentation 
     * of specific functions and members for which Mechanica 
     * automatically handles. 
     * 
     * If engine is not on a device, then the call is ignored. 
     * 
     * @return HRESULT 
     */
    HRESULT refresh();

    /**
     * @brief Set the seed for the random number generator on a CUDA device. 
     * 
     * The seed is uniformly applied to all devices. 
     * Resets all random number generators. 
     * Can be used when on or off a CUDA device. 
     * 
     * @param seed The seed. 
     * @return HRESULT 
     */
    HRESULT setSeed(const unsigned int seed);

    /**
     * @brief Get the seed for the random number generator on a CUDA device. 
     * 
     * @return unsigned int 
     */
    unsigned int getSeed();

private:
    bool on_device;
};

#endif // SRC_CUDA_MXENGINECUDACONFIG_H_
