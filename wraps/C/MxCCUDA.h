/**
 * @file MxCCUDA.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for CUDA-accelerated features
 * @date 2022-04-07
 */

#ifndef _WRAPS_C_MXCCUDA_H_
#define _WRAPS_C_MXCCUDA_H_

#include <mx_port.h>

#include "MxCBond.h"

// Handles

/**
 * @brief Handle to a @ref MxEngineCUDAConfig instance
 * 
 */
struct CAPI_EXPORT MxEngineCUDAConfigHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxBondCUDAConfig instance
 * 
 */
struct CAPI_EXPORT MxBondCUDAConfigHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxAngleCUDAConfig instance
 * 
 */
struct CAPI_EXPORT MxAngleCUDAConfigHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxSimulatorCUDAConfig instance
 * 
 */
struct CAPI_EXPORT MxSimulatorCUDAConfigHandle {
    void *MxObj;
};


////////////////////////
// MxEngineCUDAConfig //
////////////////////////


/**
 * @brief Check whether the engine is currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_onDevice(struct MxEngineCUDAConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device running the engine. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_getDevice(struct MxEngineCUDAConfigHandle *handle, int *deviceId);

/**
 * @brief Set the id of the device for running the engine. 
 * 
 * Fails if engine is currently on a device. 
 * 
 * @param handle populated handle
 * @param deviceId device id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_setDevice(struct MxEngineCUDAConfigHandle *handle, unsigned int deviceId);

/**
 * @brief Clear configured device for the engine. 
 * 
 * Fails if engine is currently on a device. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_clearDevice(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Send engine to device. If engine is already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_toDevice(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_fromDevice(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for the current CUDA device. 
 * 
 * Throws an error if called when the engine is already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_setBlocks(struct MxEngineCUDAConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for the current CUDA device. 
 * 
 * Throws an error if called when the engine is already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_setThreads(struct MxEngineCUDAConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update potentials on a CUDA device. 
 * 
 * Useful for notifying the device that a potential has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_refreshPotentials(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Update fluxes on a CUDA device. 
 * 
 * Useful for notifying the device that a flux has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_refreshFluxes(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Update boundary conditions on a CUDA device. 
 * 
 * Useful for notifying the device that a boundary condition has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_refreshBoundaryConditions(struct MxEngineCUDAConfigHandle *handle);

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
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_refresh(struct MxEngineCUDAConfigHandle *handle);

/**
 * @brief Set the seed for the random number generator on all CUDA devices. 
 * 
 * The seed is uniformly applied to all devices. 
 * Resets all random number generators. 
 * Can be used when on or off a CUDA device. 
 * 
 * @param handle populated handle
 * @param seed The seed. 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_setSeed(struct MxEngineCUDAConfigHandle *handle, unsigned int seed);

/**
 * @brief Get the seed for the random number generator on a CUDA device. 
 * 
 * @param handle populated handle
 * @param seed The seed.
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCEngineCUDAConfig_getSeed(struct MxEngineCUDAConfigHandle *handle, unsigned int *seed);


//////////////////////
// MxBondCUDAConfig //
//////////////////////


/**
 * @brief Check whether the bonds are currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_onDevice(struct MxBondCUDAConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device designated for running bonds. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_getDevice(struct MxBondCUDAConfigHandle *handle, int *deviceId);

/**
 * @brief Set the id of the device for running bonds. 
 * 
 * Can be safely called while bonds are currently on a device. 
 * 
 * @param handle populated handle
 * @param deviceId device id
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_setDevice(struct MxBondCUDAConfigHandle *handle, unsigned int deviceId);

/**
 * @brief Send bonds to device. If bonds are already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_toDevice(struct MxBondCUDAConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_fromDevice(struct MxBondCUDAConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when the bonds are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_setBlocks(struct MxBondCUDAConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when bonds are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_setThreads(struct MxBondCUDAConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update a bond on a CUDA device. 
 * 
 * Useful for notifying the device that a bond has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bh bond to update
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_refreshBond(struct MxBondCUDAConfigHandle *handle, struct MxBondHandleHandle *bh);

/**
 * @brief Update bonds on a CUDA device. 
 * 
 * Useful for notifying the device that bonds have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bonds bonds to update
 * @param numBonds number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_refreshBonds(struct MxBondCUDAConfigHandle *handle, struct MxBondHandleHandle **bonds, unsigned int numBonds);

/**
 * @brief Update all bonds on a CUDA device. 
 * 
 * Useful for notifying the device that bonds have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBondCUDAConfig_refresh(struct MxBondCUDAConfigHandle *handle);


///////////////////////
// MxAngleCUDAConfig //
///////////////////////


/**
 * @brief Check whether the angles are currently on a device. 
 * 
 * @param handle populated handle
 * @param onDevice true if currently on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_onDevice(struct MxAngleCUDAConfigHandle *handle, bool *onDevice);

/**
 * @brief Get the id of the device designated for running angles. 
 * 
 * @param handle populated handle
 * @param deviceId device id; -1 if engine is not on a device
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_getDevice(struct MxAngleCUDAConfigHandle *handle, int *deviceId);

/**
 * @brief Send angles to device. If angles are already on device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_toDevice(struct MxAngleCUDAConfigHandle *handle);

/**
 * @brief Pull engine from device. If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_fromDevice(struct MxAngleCUDAConfigHandle *handle);

/**
 * @brief Set the number of blocks of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when the angles are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numBlocks number of blocks
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_setBlocks(struct MxAngleCUDAConfigHandle *handle, unsigned int numBlocks);

/**
 * @brief Set the number of threads of the CUDA configuration for a CUDA device. 
 * 
 * Throws an error if called when angles are already deployed to a CUDA device. 
 * 
 * @param handle populated handle
 * @param numThreads number of threads
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_setThreads(struct MxAngleCUDAConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Update a angle on a CUDA device. 
 * 
 * Useful for notifying the device that a angle has changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param bh angle to update
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_refreshAngle(struct MxAngleCUDAConfigHandle *handle, struct MxAngleHandleHandle *bh);

/**
 * @brief Update angles on a CUDA device. 
 * 
 * Useful for notifying the device that angles have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @param angles angles to update
 * @param numAngles number of bonds
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_refreshAngles(struct MxAngleCUDAConfigHandle *handle, struct MxAngleHandleHandle **angles, unsigned int numAngles);

/**
 * @brief Update all angles on a CUDA device. 
 * 
 * Useful for notifying the device that angles have changed. 
 * 
 * If engine is not on a device, then the call is ignored. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCAngleCUDAConfig_refresh(struct MxAngleCUDAConfigHandle *handle);


///////////////////////////
// MxSimulatorCUDAConfig //
///////////////////////////


/**
 * @brief Get simulator CUDA runtime interface
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulator_getCUDAConfig(struct MxSimulatorCUDAConfigHandle *handle);

/**
 * @brief Get the engine CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulatorCUDAConfig_getEngine(struct MxSimulatorCUDAConfigHandle *handle, struct MxEngineCUDAConfigHandle *itf);

/**
 * @brief Get the bond CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulatorCUDAConfig_getBonds(struct MxSimulatorCUDAConfigHandle *handle, struct MxBondCUDAConfigHandle *itf);

/**
 * @brief Get the angle CUDA runtime control interface
 * 
 * @param handle populated handle
 * @param itf control interface
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulatorCUDAConfig_getAngles(struct MxSimulatorCUDAConfigHandle *handle, struct MxAngleCUDAConfigHandle *itf);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the supported CUDA architectures of this installation
 * 
 * @param str architectures
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCUDAArchs(char **str, unsigned int *numChars);


#endif // _WRAPS_C_MXCCUDA_H_