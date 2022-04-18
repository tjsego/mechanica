/**
 * @file MxCcell_polarity.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for center model cell polarity
 * @date 2022-04-07
 */

#ifndef _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_
#define _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_

#include <mx_port.h>

#include "../../../MxCParticle.h"
#include "../../../MxCForce.h"
#include "../../../MxCPotential.h"

// Handles

struct CAPI_EXPORT PolarContactTypeEnumHandle {
    unsigned int REGULAR;
    unsigned int ISOTROPIC;
    unsigned int ANISOTROPIC;
};

/**
 * @brief Handle to a @ref PolarityForcePersistent instance
 * 
 */
struct CAPI_EXPORT PolarityForcePersistentHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxPolarityArrowData instance
 * 
 */
struct CAPI_EXPORT MxPolarityArrowDataHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxCellPolarityPotentialContact instance
 * 
 */
struct CAPI_EXPORT MxCellPolarityPotentialContactHandle {
    void *MxObj;
};


//////////////////////
// PolarContactType //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPolarContactType_init(struct PolarContactTypeEnumHandle *handle);


/////////////////////////////
// PolarityForcePersistent //
/////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param sensAB proportionality of force to AB vector
 * @param sensPCP proportionality of force to PCP vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPolarityForcePersistent_init(struct PolarityForcePersistentHandle *handle, float sensAB, float sensPCP);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPolarityForcePersistent_destroy(struct PolarityForcePersistentHandle *handle);

/**
 * @brief Cast to base force. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCPolarityForcePersistent_toBase(struct PolarityForcePersistentHandle *handle, MxForceHandle *baseHandle);

/**
 * @brief Cast from base force. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCPolarityForcePersistent_fromBase(MxForceHandle *baseHandle, struct PolarityForcePersistentHandle *handle);


////////////////////////////////////
// MxCellPolarityPotentialContact //
////////////////////////////////////


/**
 * @brief Create a polarity state dynamics and anisotropic adhesion potential
 * 
 * @param handle handle to populate
 * @param cutoff cutoff distance
 * @param couplingFlat flat interaction coefficient
 * @param couplingOrtho orthogonal interaction coefficient
 * @param couplingLateral lateral interaction coefficient
 * @param distanceCoeff distance coefficient
 * @param cType contact type (e.g., normal, isotropic or anisotropic)
 * @param mag magnitude of force due to potential
 * @param rate state vector dynamics rate due to potential
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_init(struct MxCellPolarityPotentialContactHandle *handle, 
                                                        float cutoff, 
                                                        float couplingFlat, 
                                                        float couplingOrtho, 
                                                        float couplingLateral, 
                                                        float distanceCoeff, 
                                                        unsigned int cType, 
                                                        float mag, 
                                                        float rate, 
                                                        float bendingCoeff);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_destroy(struct MxCellPolarityPotentialContactHandle *handle);

/**
 * @brief Cast to base potential. 
 * 
 * @param handle populated handle
 * @param baseHandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_toBase(struct MxCellPolarityPotentialContactHandle *handle, MxPotentialHandle *baseHandle);

/**
 * @brief Cast from base potential. Fails if instance is not of this type
 * 
 * @param baseHandle populated handle 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_fromBase(struct MxPotentialHandle *baseHandle, struct MxCellPolarityPotentialContactHandle *handle);

/**
 * @brief Get the flat interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingFlat flat interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getCouplingFlat(struct MxCellPolarityPotentialContactHandle *handle, float *couplingFlat);

/**
 * @brief Set the flat interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingFlat flat interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setCouplingFlat(struct MxCellPolarityPotentialContactHandle *handle, float couplingFlat);

/**
 * @brief Get the orthogonal interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingOrtho orthogonal interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getCouplingOrtho(struct MxCellPolarityPotentialContactHandle *handle, float *couplingOrtho);

/**
 * @brief Set the orthogonal interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingOrtho orthogonal interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setCouplingOrtho(struct MxCellPolarityPotentialContactHandle *handle, float couplingOrtho);

/**
 * @brief Get the lateral interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingLateral lateral interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getCouplingLateral(struct MxCellPolarityPotentialContactHandle *handle, float *couplingLateral);

/**
 * @brief Set the lateral interaction coefficient
 * 
 * @param handle populated handle
 * @param couplingLateral lateral interaction coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setCouplingLateral(struct MxCellPolarityPotentialContactHandle *handle, float couplingLateral);

/**
 * @brief Get the distance coefficient
 * 
 * @param handle populated handle
 * @param distanceCoeff distance coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getDistanceCoeff(struct MxCellPolarityPotentialContactHandle *handle, float *distanceCoeff);

/**
 * @brief Set the distance coefficient
 * 
 * @param handle populated handle
 * @param distanceCoeff distance coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setDistanceCoeff(struct MxCellPolarityPotentialContactHandle *handle, float distanceCoeff);

/**
 * @brief Get the contact type (e.g., normal, isotropic or anisotropic)
 * 
 * @param handle populated handle
 * @param cType contact type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getCType(struct MxCellPolarityPotentialContactHandle *handle, unsigned int *cType);

/**
 * @brief Set the contact type (e.g., normal, isotropic or anisotropic)
 * 
 * @param handle populated handle
 * @param cType contact type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setCType(struct MxCellPolarityPotentialContactHandle *handle, unsigned int cType);

/**
 * @brief Get the magnitude of force due to potential
 * 
 * @param handle populated handle
 * @param mag magnitude of force due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getMag(struct MxCellPolarityPotentialContactHandle *handle, float *mag);

/**
 * @brief Set the magnitude of force due to potential
 * 
 * @param handle populated handle
 * @param mag magnitude of force due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setMag(struct MxCellPolarityPotentialContactHandle *handle, float mag);

/**
 * @brief Get the state vector dynamics rate due to potential
 * 
 * @param handle populated handle
 * @param rate state vector dynamics rate due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getRate(struct MxCellPolarityPotentialContactHandle *handle, float *rate);

/**
 * @brief Set the state vector dynamics rate due to potential
 * 
 * @param handle populated handle
 * @param rate state vector dynamics rate due to potential
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setRate(struct MxCellPolarityPotentialContactHandle *handle, float rate);

/**
 * @brief Get the bending coefficient
 * 
 * @param handle populated handle
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_getBendingCoeff(struct MxCellPolarityPotentialContactHandle *handle, float *bendingCoeff);

/**
 * @brief Set the bending coefficient
 * 
 * @param handle populated handle
 * @param bendingCoeff bending coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarityPotentialContact_setBendingCoeff(struct MxCellPolarityPotentialContactHandle *handle, float bendingCoeff);


/////////////////////////
// MxPolarityArrowData //
/////////////////////////


/**
 * @brief Get the arrow length
 * 
 * @param handle populated handle
 * @param arrowLength arrow length
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPolarityArrowData_getArrowLength(struct MxPolarityArrowDataHandle *handle, float *arrowLength);

/**
 * @brief Set the arrow length
 * 
 * @param handle populated handle
 * @param arrowLength arrow length
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPolarityArrowData_setArrowLength(struct MxPolarityArrowDataHandle *handle, float arrowLength);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Gets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @param vec polarity vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetVectorAB(int pId, bool current, float **vec);

/**
 * @brief Gets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param current current value flag; default true
 * @param vec polarity vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetVectorPCP(int pId, bool current, float **vec);

/**
 * @brief Sets the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; set to true to set the current value
 * @param init initialization flag; set to true to set the initial value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetVectorAB(int pId, float *pVec, bool current, bool init);

/**
 * @brief Sets the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param pVec vector value
 * @param current current value flag; set to true to set the current value
 * @param init initialization flag; set to true to set the initial value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetVectorPCP(int pId, float *pVec, bool current, bool init);

/**
 * @brief Updates all running polarity models
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_update();

/**
 * @brief Registers a particle as polar. 
 * 
 * This must be called before the first integration step.
 * Otherwise, the engine will not know that the particle 
 * is polar and will be ignored. 
 * 
 * @param ph handle of particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_registerParticle(struct MxParticleHandleHandle *ph);

/**
 * @brief Unregisters a particle as polar. 
 * 
 * This must be called before destroying a registered particle. 
 * 
 * @param ph handle of particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_unregister(struct MxParticleHandleHandle *ph);

/**
 * @brief Registers a particle type as polar. 
 * 
 * This must be called on a particle type before any other type-specific operations. 
 * 
 * @param pType particle type
 * @param initMode initialization mode for particles of this type
 * @param initPolarAB initial value of AB polarity vector; only used when initMode="value"
 * @param initPolarPCP initial value of PCP polarity vector; only used when initMode="value"
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_registerType(struct MxParticleTypeHandle *pType, const char *initMode, float *initPolarAB, float *initPolarPCP);

/**
 * @brief Gets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param initMode initialization mode
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetInitMode(struct MxParticleTypeHandle *pType, char **initMode, unsigned int *numChars);

/**
 * @brief Sets the name of the initialization mode of a type
 * 
 * @param pType a type
 * @param value initialization mode
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetInitMode(struct MxParticleTypeHandle *pType, const char *value);

/**
 * @brief Gets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param vec vector value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetInitPolarAB(struct MxParticleTypeHandle *pType, float **vec);

/**
 * @brief Sets the initial AB polar vector of a type
 * 
 * @param pType a type
 * @param value initial AB polar vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetInitPolarAB(struct MxParticleTypeHandle *pType, float *value);

/**
 * @brief Gets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param vec initial PCP polar vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetInitPolarPCP(struct MxParticleTypeHandle *pType, float **vec);

/**
 * @brief Sets the initial PCP polar vector of a type
 * 
 * @param pType a type
 * @param value initial PCP polar vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetInitPolarPCP(struct MxParticleTypeHandle *pType, float *value);

/**
 * @brief Toggles whether polarity vectors are rendered
 * 
 * @param _draw rendering flag; vectors are rendered when true
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetDrawVectors(bool _draw);

/**
 * @brief Sets rendered polarity vector colors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param colorAB name of AB vector color
 * @param colorPCP name of PCP vector color
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetArrowColors(const char *colorAB, const char *colorPCP);

/**
 * @brief Sets scale of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _scale scale of rendered vectors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetArrowScale(float _scale);

/**
 * @brief Sets length of rendered polarity vectors. 
 * 
 * Applies to subsequently created vectors and all current vectors. 
 * 
 * @param _length length of rendered vectors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_SetArrowLength(float _length);

/**
 * @brief Gets the rendering info for the AB polarity vector of a cell
 * 
 * @param pId particle id
 * @param arrowData rendering info
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetVectorArrowAB(unsigned int pId, struct MxPolarityArrowDataHandle *arrowData);

/**
 * @brief Gets the rendering info for the PCP polarity vector of a cell
 * 
 * @param pId particle id
 * @param arrowData rendering info
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_GetVectorArrowPCP(unsigned int pId, struct MxPolarityArrowDataHandle *arrowData);

/**
 * @brief Runs the polarity model along with a simulation. 
 * Must be called before doing any operations with this module. 
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCellPolarity_load();

#endif // _WRAPS_C_MODELS_CENTER_CELL_POLARITY_CELL_POLARITY_H_