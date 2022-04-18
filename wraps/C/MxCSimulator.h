/**
 * @file MxCSimulator.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSimulator
 * @date 2022-03-24
 */

#ifndef _WRAPS_C_MXCSIMULATOR_H_
#define _WRAPS_C_MXCSIMULATOR_H_

#include <mx_port.h>

#include "MxCUniverse.h"

// Handles

struct CAPI_EXPORT MxSimulator_EngineIntegratorHandle {
    int FORWARD_EULER;
    int RUNGE_KUTTA_4;
};

struct CAPI_EXPORT MxSimulator_DpiScalingPolicyHandle {
    int MXSIMULATOR_NONE;
    int MXSIMULATOR_WINDOWLESS;
    int MXSIMULATOR_GLFW;
};

/**
 * @brief Handle to a @ref MxSimulator_Config instance
 * 
 */
struct CAPI_EXPORT MxSimulator_ConfigHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxSimulator instance
 * 
 */
struct CAPI_EXPORT MxSimulatorHandle {
    void *MxObj;
};


//////////////////////////////////
// MxSimulator_EngineIntegrator //
//////////////////////////////////


/**
 * @brief Populate engine integrator enums
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_EngineIntegrator_init(struct MxSimulator_EngineIntegratorHandle *handle);


//////////////////////////////////
// MxSimulator_DpiScalingPolicy //
//////////////////////////////////


/**
 * @brief Populate dpi scaling policy enums
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_DpiScalingPolicy_init(struct MxSimulator_DpiScalingPolicyHandle *handle);


////////////////////////
// MxSimulator_Config //
////////////////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_init(struct MxSimulator_ConfigHandle *handle);

/**
 * @brief Get the title of a configuration
 * 
 * @param handle populated handle
 * @param title title
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getTitle(struct MxSimulator_ConfigHandle *handle, char **title, unsigned int *numChars);

/**
 * @brief Set the title of a configuration
 * 
 * @param handle populated handle
 * @param title title
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_setTitle(struct MxSimulator_ConfigHandle *handle, const char *title);

/**
 * @brief Get the window size
 * 
 * @param handle populated handle
 * @param x width
 * @param y height
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getWindowSize(struct MxSimulator_ConfigHandle *handle, unsigned int *x, unsigned int *y);

/**
 * @brief Set the window size
 * 
 * @param handle populated handle
 * @param x width
 * @param y height
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_setWindowSize(struct MxSimulator_ConfigHandle *handle, unsigned int x, unsigned int y);

/**
 * @brief Get the random number generator seed. If none is set, returns NULL.
 * 
 * @param handle populated handle
 * @param seed random number generator seed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getSeed(struct MxSimulator_ConfigHandle *handle, unsigned int *seed);

/**
 * @brief Set the random number generator seed.
 * 
 * @param handle populated handle
 * @param seed random number generator seed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_setSeed(struct MxSimulator_ConfigHandle *handle, unsigned int seed);

/**
 * @brief Get the windowless flag
 * 
 * @param handle populated handle
 * @param windowless windowless flag
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getWindowless(struct MxSimulator_ConfigHandle *handle, bool *windowless);

/**
 * @brief Set the windowless flag
 * 
 * @param handle populated handle
 * @param windowless windowless flag
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_setWindowless(struct MxSimulator_ConfigHandle *handle, bool windowless);

/**
 * @brief Get the imported data file path during initialization, if any.
 * 
 * @param handle populated handle
 * @param filePath file path
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getImportDataFilePath(struct MxSimulator_ConfigHandle *handle, char **filePath, unsigned int *numChars);

/**
 * @brief Get the current clip planes
 * 
 * @param handle populated handle
 * @param clipPlanes clip planes
 * @param numClipPlanes number of clip planes
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getClipPlanes(struct MxSimulator_ConfigHandle *handle, float **clipPlanes, unsigned int *numClipPlanes);

/**
 * @brief Set the clip planes
 * 
 * @param handle populated handle
 * @param clipPlanes clip planes
 * @param numClipPlanes number of clip planes
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_setClipPlanes(struct MxSimulator_ConfigHandle *handle, float *clipPlanes, unsigned int numClipPlanes);

/**
 * @brief Get the universe configuration
 * 
 * @param handle populated handle
 * @param confHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_getUniverseConfig(struct MxSimulator_ConfigHandle *handle, struct MxUniverseConfigHandle *confHandle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_Config_destroy(struct MxSimulator_ConfigHandle *handle);


/////////////////
// MxSimulator //
/////////////////

/**
 * @brief Main simulator init method
 * 
 * @param argv initializer arguments
 * @param nargs number of arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_init(const char **argv, unsigned int nargs);

/**
 * @brief Main simulator init method
 * 
 * @param conf configuration
 * @param appArgv app arguments
 * @param nargs number of app arguments
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_initC(struct MxSimulator_ConfigHandle *conf, const char **appArgv, unsigned int nargs);

/**
 * @brief Gets the global simulator object
 * 
 * @param handle handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_get(struct MxSimulatorHandle *handle);

/**
 * @brief Make the instance the global simulator object
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSimulator_makeCurrent(struct MxSimulatorHandle *handle);

/**
 * @brief Runs the event loop until all windows close or simulation time expires. 
 * Automatically performs universe time propogation. 
 * 
 * @param et final time; a negative number runs infinitely
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_run(double et);

/**
 * @brief Shows any windows that were specified in the config. 
 * 
 * Does not start the universe time propagation unlike @ref MxCSimulator_run().
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_show();

/**
 * @brief Closes the main window, while the application / simulation continues to run.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_close();

/**
 * @brief Destroy the simulation.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_destroy();

/**
 * @brief Issue call to rendering update.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_redraw();

/**
 * @brief Get the number of threads
 * 
 * @param numThreads number of threads
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCSimulator_getNumThreads(unsigned int *numThreads);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Test whether running interactively
 * 
 */
CAPI_FUNC(bool) MxC_TerminalInteractiveShell();

/**
 * @brief Set whether running interactively
 */
CAPI_FUNC(HRESULT) MxC_setTerminalInteractiveShell(bool _interactive);

#endif // _WRAPS_C_MXCSIMULATOR_H_