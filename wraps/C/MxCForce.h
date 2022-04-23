/**
 * @file MxCForce.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxForce
 * @date 2022-03-30
 */

#ifndef _WRAPS_C_MXCFORCE_H_
#define _WRAPS_C_MXCFORCE_H_

#include <mx_port.h>

typedef void (*MxUserForceFuncTypeHandleFcn)(struct MxConstantForceHandle*, float*);

// Handles

/**
 * @brief Handle to a @ref MXFORCE_TYPE instance
 * 
 */
struct CAPI_EXPORT MXFORCE_TYPEHandle {
    unsigned int FORCE_FORCE;
    unsigned int FORCE_BERENDSEN;
    unsigned int FORCE_GAUSSIAN;
    unsigned int FORCE_FRICTION;
    unsigned int FORCE_SUM;
    unsigned int FORCE_CONSTANT;
};

/**
 * @brief Handle to a @ref MxUserForceFuncType instance
 * 
 */
struct CAPI_EXPORT MxUserForceFuncTypeHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxForce instance
 * 
 */
struct CAPI_EXPORT MxForceHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxForceSum instance
 * 
 */
struct CAPI_EXPORT MxForceSumHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxConstantForce instance
 * 
 */
struct CAPI_EXPORT MxConstantForceHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Berendsen instance
 * 
 */
struct CAPI_EXPORT BerendsenHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Gaussian instance
 * 
 */
struct CAPI_EXPORT GaussianHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref Friction instance
 * 
 */
struct CAPI_EXPORT FrictionHandle {
    void *MxObj;
};


//////////////////
// MXFORCE_TYPE //
//////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFORCE_TYPE_init(struct MXFORCE_TYPEHandle *handle);


/////////////////////////
// MxUserForceFuncType //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param fcn evaluation function
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_EvalFcn_init(struct MxUserForceFuncTypeHandle *handle, MxUserForceFuncTypeHandleFcn *fcn);

/**
 * @brief Destroy an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_EvalFcn_destroy(struct MxUserForceFuncTypeHandle *handle);


/////////////
// MxForce //
/////////////

/**
 * @brief Get the force type
 * 
 * @param handle populated handle
 * @param te type enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_getType(struct MxForceHandle *handle, unsigned int *te);

/**
 * @brief Bind a force to a species. 
 * 
 * When a force is bound to a species, the magnitude of the force is scaled by the concentration of the species. 
 * 
 * @param handle populated handle
 * @param a_type particle type containing the species
 * @param coupling_symbol symbol of the species
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_bind_species(struct MxForceHandle *handle, struct MxParticleTypeHandle *a_type, const char *coupling_symbol);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_toString(struct MxForceHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * The returned type is automatically registered with the engine. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_fromString(struct MxForceHandle *handle, const char *str);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_destroy(struct MxForceHandle *handle);


////////////////
// MxForceSum //
////////////////


/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForceSum_checkType(struct MxForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForceSum_toBase(struct MxForceSumHandle *handle, struct MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForceSum_fromBase(struct MxForceHandle *baseHandle, struct MxForceSumHandle *handle);

/**
 * @brief Get the constituent forces
 * 
 * @param handle populated handle
 * @param f1 first constituent force
 * @param f2 second constituent force
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForceSum_getConstituents(struct MxForceSumHandle *handle, struct MxForceHandle *f1, struct MxForceHandle *f2);


/////////////////////
// MxConstantForce //
/////////////////////


/**
 * @brief Create a constant force with a force function
 * 
 * @param handle handle to populate
 * @param func function to evaluate the force components
 * @param period force period; infinite if a negative value is passed
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_init(struct MxConstantForceHandle *handle, struct MxUserForceFuncTypeHandle *func, float period);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_checkType(struct MxForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_toBase(struct MxConstantForceHandle *handle, struct MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_fromBase(struct MxForceHandle *baseHandle, struct MxConstantForceHandle *handle);

/**
 * @brief Get force period
 * 
 * @param handle populated handle 
 * @param period force period
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_getPeriod(struct MxConstantForceHandle *handle, float *period);

/**
 * @brief Set force period
 * 
 * @param handle populated handle 
 * @param period force period
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_setPeriod(struct MxConstantForceHandle *handle, float period);

/**
 * @brief Set force function
 * 
 * @param handle populated handle 
 * @param fcn force function
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_setFunction(struct MxConstantForceHandle *handle, struct MxUserForceFuncTypeHandle *fcn);

/**
 * @brief Get current force value
 * 
 * @param handle populated handle 
 * @param force force value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_getValue(struct MxConstantForceHandle *handle, float **force);

/**
 * @brief Set current force value
 * 
 * @param handle populated handle 
 * @param force force value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_setValue(struct MxConstantForceHandle *handle, float *force);

/**
 * @brief Get time of last force update
 * 
 * @param handle populated handle 
 * @param lastUpdate time of last force update
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCConstantForce_getLastUpdate(struct MxConstantForceHandle *handle, float *lastUpdate);


///////////////
// Berendsen //
///////////////

/**
 * @brief Creates a Berendsen thermostat. 
 * 
 * The thermostat uses the target temperature @f$ T_0 @f$ from the object 
 * to which it is bound. 
 * The Berendsen thermostat effectively re-scales the velocities of an object in 
 * order to make the temperature of that family of objects match a specified 
 * temperature.
 * 
 * The Berendsen thermostat force has the function form: 
 * 
 * @f[
 * 
 *      \frac{\mathbf{p}}{\tau_T} \left(\frac{T_0}{T} - 1 \right),
 * 
 * @f]
 * 
 * where @f$ \mathbf{p} @f$ is the momentum, 
 * @f$ T @f$ is the measured temperature of a family of 
 * particles, @f$ T_0 @f$ is the control temperature, and 
 * @f$ \tau_T @f$ is the coupling constant. The coupling constant is a measure 
 * of the time scale on which the thermostat operates, and has units of 
 * time. Smaller values of @f$ \tau_T @f$ result in a faster acting thermostat, 
 * and larger values result in a slower acting thermostat.
 * 
 * @param handle handle to populate
 * @param tau time constant that determines how rapidly the thermostat effects the system.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_init(struct BerendsenHandle *handle, float tau);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_checkType(struct MxForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_toBase(struct BerendsenHandle *handle, struct MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_fromBase(struct MxForceHandle *baseHandle, struct BerendsenHandle *handle);

/**
 * @brief Get the time constant
 * 
 * @param handle populated handle
 * @param tau time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_getTimeConstant(struct BerendsenHandle *handle, float *tau);

/**
 * @brief Set the time constant
 * 
 * @param handle populated handle
 * @param tau time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBerendsen_setTimeConstant(struct BerendsenHandle *handle, float tau);


//////////////
// Gaussian //
//////////////

/**
 * @brief Creates a random force. 
 * 
 * A random force has a randomly selected orientation and magnitude. 
 * 
 * Orientation is selected according to a uniform distribution on the unit sphere. 
 * 
 * Magnitude is selected according to a prescribed mean and standard deviation. 
 * 
 * @param handle handle to populate
 * @param std standard deviation of magnitude
 * @param mean mean of magnitude
 * @param duration duration of force. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_init(struct GaussianHandle *handle, float std, float mean, float duration);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_checkType(struct MxForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_toBase(struct GaussianHandle *handle, struct MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_fromBase(struct MxForceHandle *baseHandle, struct GaussianHandle *handle);

/**
 * @brief Get the magnitude standard deviation
 * 
 * @param handle populated handle
 * @param std magnitude standard deviation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_getStd(struct GaussianHandle *handle, float *std);

/**
 * @brief Set the magnitude standard deviation
 * 
 * @param handle populated handle
 * @param std magnitude standard deviation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_setStd(struct GaussianHandle *handle, float std);

/**
 * @brief Get the magnitude mean
 * 
 * @param handle populated handle
 * @param mean mean magnitude
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_getMean(struct GaussianHandle *handle, float *mean);

/**
 * @brief Set the magnitude mean
 * 
 * @param handle populated handle
 * @param mean mean magnitude
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_setMean(struct GaussianHandle *handle, float mean);

/**
 * @brief Get the magnitude duration
 * 
 * @param handle populated handle
 * @param duration magnitude duration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_getDuration(struct GaussianHandle *handle, float *duration);

/**
 * @brief Set the magnitude duration
 * 
 * @param handle populated handle
 * @param duration magnitude duration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCGaussian_setDuration(struct GaussianHandle *handle, float duration);


//////////////
// Friction //
//////////////


/**
 * @brief Creates a friction force. 
 * 
 * A friction force has the form: 
 * 
 * @f[
 * 
 *      - \frac{|| \mathbf{v} ||}{\tau} \mathbf{v} ,
 * 
 * @f]
 * 
 * where @f$ \mathbf{v} @f$ is the velocity of a particle and @f$ \tau @f$ is a time constant. 
 * 
 * @param handle handle to populate
 * @param coef time constant
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_init(struct FrictionHandle *handle, float coeff);

/**
 * @brief Check whether a base handle is of this force type
 * 
 * @param handle populated handle
 * @param isType flag signifying whether the handle is of this force type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_checkType(struct MxForceHandle *handle, bool *isType);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_toBase(struct FrictionHandle *handle, struct MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_fromBase(struct MxForceHandle *baseHandle, struct FrictionHandle *handle);

/**
 * @brief Get the friction coefficient
 * 
 * @param handle populated handle
 * @param coef friction coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_getCoef(struct FrictionHandle *handle, float *coef);

/**
 * @brief Set the friction coefficient
 * 
 * @param handle populated handle
 * @param coef friction coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFriction_setCoef(struct FrictionHandle *handle, float coef);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Add two forces
 * 
 * @param f1 first force
 * @param f2 second force
 * @param fSum handle to populate with force sum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCForce_add(struct MxForceHandle *f1, struct MxForceHandle *f2, struct MxForceSumHandle *fSum);


#endif // _WRAPS_C_MXCFORCE_H_