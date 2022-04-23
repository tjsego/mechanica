/**
 * @file MxCUniverse.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxUniverse
 * @date 2022-03-28
 */

#ifndef _WRAPS_C_MXCUNIVERSE_H_
#define _WRAPS_C_MXCUNIVERSE_H_

#include <mx_port.h>

#include "MxCParticle.h"
#include "MxCBoundaryConditions.h"

// Handles

/**
 * @brief Handle to a @ref MxUniverseConfig instance
 * 
 */
struct CAPI_EXPORT MxUniverseConfigHandle {
    void *MxObj;
};


////////////////
// MxUniverse //
////////////////


/**
 * @brief Get the origin of the universe
 * 
 * @param origin 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getOrigin(float **origin);

/**
 * @brief Get the dimensions of the universe
 * 
 * @param dim 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getDim(float **dim);

/**
 * @brief Get whether the universe is running
 * 
 * @param isRunning 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getIsRunning(bool *isRunning);

/**
 * @brief Get the name of the model / script
 * 
 * @param name 
 * @param numChars 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getName(char **name, unsigned int *numChars);

/**
 * @brief Get the virial tensor of the universe
 * 
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getVirial(float **virial);

/**
 * @brief Get the virial tensor of the universe for a set of particle types
 * 
 * @param phandles array of types
 * @param numTypes number of types
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getVirialT(struct MxParticleTypeHandle **phandles, unsigned int numTypes, float **virial);

/**
 * @brief Get the virial tensor of a neighborhood
 * 
 * @param origin origin of neighborhood
 * @param numTypes radius of neighborhood
 * @param virial virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getVirialO(float *origin, float radius, float **virial);

/**
 * @brief Get the virial tensor for a set of particle types in a neighborhood
 * 
 * @param phandles array of types
 * @param numTypes number of types
 * @param origin origin of neighborhood
 * @param numTypes radius of neighborhood
 * @param virial virial tensor
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCUniverse_getVirialOT(struct MxParticleTypeHandle **phandles, 
                                           unsigned int numTypes, 
                                           float *origin, 
                                           float radius, 
                                           float **virial);

/**
 * @brief Get the number of particles in the universe
 * 
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getNumParts(unsigned int *numParts);

/**
 * @brief Get the i'th particle of the universe
 * 
 * @param pidx index of particle
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getParticle(unsigned int pidx, struct MxParticleHandleHandle *handle);

/**
 * @brief Get the center of the universe
 * 
 * @param center 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getCenter(float **center);

/**
 * @brief Integrates the universe for a duration as given by ``until``, or for a single time step 
 * if 0 is passed.
 * 
 * @param until runs the timestep for this length of time.
 * @param dt overrides the existing time step, and uses this value for time stepping; currently not supported.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_step(double until, double dt);

/**
 * @brief Stops the universe time evolution. This essentially freezes the universe, 
 * everything remains the same, except time no longer moves forward.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_stop();

/**
 * @brief Starts the universe time evolution, and advanced the universe forward by 
 * timesteps in ``dt``. All methods to build and manipulate universe objects 
 * are valid whether the universe time evolution is running or stopped.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_start();

/**
 * @brief Reset the universe
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_reset();

/**
 * @brief Reset all species in all particles
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_resetSpecies();

/**
 * @brief Get the universe temperature. 
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param temperature 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getTemperature(double *temperature);

/**
 * @brief Get the current time
 * 
 * @param time 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getTime(double *time);

/**
 * @brief Get the period of a time step
 * 
 * @param dt 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getDt(double *dt);

/**
 * @brief Get the boundary conditions
 * 
 * @param bcs boundary conditions 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getBoundaryConditions(struct MxBoundaryConditionsHandle *bcs);

/**
 * @brief Get the current system kinetic energy
 * 
 * @param ke 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getKineticEnergy(double *ke);

/**
 * @brief Get the current number of registered particle types
 * 
 * @param numTypes 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getNumTypes(int *numTypes);

/**
 * @brief Get the global interaction cutoff distance
 * 
 * @param cutoff 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverse_getCutoff(double *cutoff);


////////////////////////////
// MxUniverseConfigHandle //
////////////////////////////


/**
 * @brief Initialize a new instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_init(struct MxUniverseConfigHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_destroy(struct MxUniverseConfigHandle *handle);

/**
 * @brief Get the origin of the universe
 * 
 * @param handle populated handle
 * @param origin 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getOrigin(struct MxUniverseConfigHandle *handle, float **origin);

/**
 * @brief Set the origin of the universe
 * 
 * @param handle populated handle
 * @param origin 3-element array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setOrigin(struct MxUniverseConfigHandle *handle, float *origin);

/**
 * @brief Get the dimensions of the universe
 * 
 * @param handle populated handle
 * @param dim 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getDim(struct MxUniverseConfigHandle *handle, float **dim);

/**
 * @brief Set the dimensions of the universe
 * 
 * @param handle populated handle
 * @param dim 3-element array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setDim(struct MxUniverseConfigHandle *handle, float *dim);

/**
 * @brief Get the grid discretization
 * 
 * @param handle populated handle
 * @param cells grid discretization
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getCells(struct MxUniverseConfigHandle *handle, int **cells);

/**
 * @brief Set the grid discretization
 * 
 * @param handle populated handle
 * @param cells grid discretization
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setCells(struct MxUniverseConfigHandle *handle, int *cells);

/**
 * @brief Get the global interaction cutoff distance
 * 
 * @param handle populated handle
 * @param cutoff cutoff distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getCutoff(struct MxUniverseConfigHandle *handle, double *cutoff);

/**
 * @brief Set the global interaction cutoff distance
 * 
 * @param handle populated handle
 * @param cutoff cutoff distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setCutoff(struct MxUniverseConfigHandle *handle, double cutoff);

/**
 * @brief Get the universe flags
 * 
 * @param handle populated handle
 * @param flags universe flags
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getFlags(struct MxUniverseConfigHandle *handle, unsigned int *flags);

/**
 * @brief Set the universe flags
 * 
 * @param handle populated handle
 * @param flags universe flags
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setFlags(struct MxUniverseConfigHandle *handle, unsigned int flags);

/**
 * @brief Get the period of a time step
 * 
 * @param handle populated handle
 * @param dt period of a time step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getDt(struct MxUniverseConfigHandle *handle, double *dt);

/**
 * @brief Set the period of a time step
 * 
 * @param handle populated handle
 * @param dt period of a time step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setDt(struct MxUniverseConfigHandle *handle, double dt);

/**
 * @brief Get the universe temperature. 
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param handle populated handle
 * @param temperature universe temperature
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getTemperature(struct MxUniverseConfigHandle *handle, double *temperature);

/**
 * @brief Set the universe temperature.
 * 
 * The universe can be run with, or without a thermostat. With a thermostat, 
 * getting / setting the temperature changes the temperature that the thermostat 
 * will try to keep the universe at. When the universe is run without a 
 * thermostat, reading the temperature returns the computed universe temp, but 
 * attempting to set the temperature yields an error. 
 * 
 * @param handle populated handle
 * @param temperature universe temperature
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setTemperature(struct MxUniverseConfigHandle *handle, double temperature);

/**
 * @brief Get the number of threads for parallel execution.
 * 
 * @param handle populated handle
 * @param numThreads number of threads for parallel execution
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getNumThreads(struct MxUniverseConfigHandle *handle, unsigned int *numThreads);

/**
 * @brief Set the number of threads for parallel execution.
 * 
 * @param handle populated handle
 * @param numThreads number of threads for parallel execution
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setNumThreads(struct MxUniverseConfigHandle *handle, unsigned int numThreads);

/**
 * @brief Get the engine integrator enum.
 * 
 * @param handle populated handle
 * @param integrator engine integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getIntegrator(struct MxUniverseConfigHandle *handle, unsigned int *integrator);

/**
 * @brief Set the engine integrator enum.
 * 
 * @param handle populated handle
 * @param integrator engine integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setIntegrator(struct MxUniverseConfigHandle *handle, unsigned int integrator);

/**
 * @brief Get the boundary condition argument container
 * 
 * @param handle populated handle
 * @param bargsHandle argument container
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_getBoundaryConditions(struct MxUniverseConfigHandle *handle, struct MxBoundaryConditionsArgsContainerHandle *bargsHandle);

/**
 * @brief Set the boundary condition argument container
 * 
 * @param handle populated handle
 * @param bargsHandle argument container
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCUniverseConfig_setBoundaryConditions(struct MxUniverseConfigHandle *handle, struct MxBoundaryConditionsArgsContainerHandle *bargsHandle);

#endif // _WRAPS_C_MXCUNIVERSE_H_