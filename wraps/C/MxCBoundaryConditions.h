/**
 * @file MxCBoundaryConditions.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBoundaryConditions
 * @date 2022-03-31
 */

#ifndef _WRAPS_C_MXCBOUNDARYCONDITIONS_H_
#define _WRAPS_C_MXCBOUNDARYCONDITIONS_H_

#include <mx_port.h>

#include "MxCParticle.h"
#include "MxCPotential.h"

// Handles

/**
 * @brief Enums for kind of boundary conditions along directions
 * 
 */
struct CAPI_EXPORT BoundaryConditionSpaceKindHandle {
    unsigned int SPACE_PERIODIC_X;
    unsigned int SPACE_PERIODIC_Y;
    unsigned int SPACE_PERIODIC_Z;
    unsigned int SPACE_PERIODIC_FULL;
    unsigned int SPACE_FREESLIP_X;
    unsigned int SPACE_FREESLIP_Y;
    unsigned int SPACE_FREESLIP_Z;
    unsigned int SPACE_FREESLIP_FULL;
};

/**
 * @brief Enums for kind of individual boundary condition
 * 
 */
struct CAPI_EXPORT BoundaryConditionKindHandle {
    unsigned int BOUNDARY_PERIODIC;
    unsigned int BOUNDARY_FREESLIP;
    unsigned int BOUNDARY_RESETTING;
};

/**
 * @brief Handle to a @ref MxBoundaryCondition instance
 * 
 */
struct CAPI_EXPORT MxBoundaryConditionHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxBoundaryConditions instance
 * 
 */
struct CAPI_EXPORT MxBoundaryConditionsHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxBoundaryConditionsArgsContainer instance
 * 
 */
struct CAPI_EXPORT MxBoundaryConditionsArgsContainerHandle {
    void *MxObj;
};


////////////////////////////////
// BoundaryConditionSpaceKind //
////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return HRESULT S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionSpaceKind_init(struct BoundaryConditionSpaceKindHandle *handle);


///////////////////////////
// BoundaryConditionKind //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionKind_init(struct BoundaryConditionKindHandle *handle);


/////////////////////////
// MxBoundaryCondition //
/////////////////////////


/**
 * @brief Get the id of a boundary condition
 * 
 * @param handle populated handle
 * @param bid boundary condition id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_getId(struct MxBoundaryConditionHandle *handle, int *bid);

/**
 * @brief Get the velocity of a boundary condition
 * 
 * @param handle populated handle
 * @param velocity 3-element allocated array of velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_getVelocity(struct MxBoundaryConditionHandle *handle, float **velocity);

/**
 * @brief Set the velocity of a boundary condition
 * 
 * @param handle populated handle
 * @param velocity 3-element allocated array of velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_setVelocity(struct MxBoundaryConditionHandle *handle, float *velocity);

/**
 * @brief Get the restore coefficient of a boundary condition
 * 
 * @param handle populated handle
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_getRestore(struct MxBoundaryConditionHandle *handle, float *restore);

/**
 * @brief Set the restore coefficient of a boundary condition
 * 
 * @param handle populated handle
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_setRestore(struct MxBoundaryConditionHandle *handle, float restore);

/**
 * @brief Get the normal of a boundary condition
 * 
 * @param handle populated handle
 * @param normal 3-element allocated array of normal
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_getNormal(struct MxBoundaryConditionHandle *handle, float **normal);

/**
 * @brief Get the equivalent radius of a boundary condition
 * 
 * @param handle populated handle
 * @param radius equivalent radius
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_getRadius(struct MxBoundaryConditionHandle *handle, float *radius);

/**
 * @brief Set the equivalent radius of a boundary condition
 * 
 * @param handle populated handle
 * @param radius equivalent radius
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_setRadius(struct MxBoundaryConditionHandle *handle, float radius);

/**
 * @brief Set the potential of a boundary condition for a particle type
 * 
 * @param handle populated handle
 * @param partHandle particle type
 * @param potHandle boundary potential
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryCondition_setPotential(struct MxBoundaryConditionHandle *handle, struct MxParticleTypeHandle *partHandle, struct MxPotentialHandle *potHandle);


//////////////////////////
// MxBoundaryConditions //
//////////////////////////


/**
 * @brief Get the top boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getTop(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Get the bottom boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getBottom(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Get the left boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getLeft(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Get the right boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getRight(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Get the front boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getFront(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Get the back boundary condition
 * 
 * @param handle populated handle
 * @param bchandle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_getBack(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle);

/**
 * @brief Set the potential on all boundaries for a particle type
 * 
 * @param handle populated handle
 * @param partHandle particle type
 * @param potHandle potential
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditions_setPotential(struct MxBoundaryConditionsHandle *handle, struct MxParticleTypeHandle *partHandle, struct MxPotentialHandle *potHandle);


///////////////////////////////////////
// MxBoundaryConditionsArgsContainer //
///////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_init(struct MxBoundaryConditionsArgsContainerHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_destroy(struct MxBoundaryConditionsArgsContainerHandle *handle);

/**
 * @brief Test whether a value type is applied to all boundaries
 * 
 * @param handle populated handle
 * @param has flag for whether a value type is applied to all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_hasValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, bool *has);

/**
 * @brief Get the boundary type value on all boundaries
 * 
 * @param handle populated handle
 * @param _bcValue boundary type value on all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_getValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, unsigned int *_bcValue);

/**
 * @brief Set the boundary type value on all boundaries
 * 
 * @param handle populated handle
 * @param _bcValue boundary type value on all boundaries
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_setValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, unsigned int _bcValue);

/**
 * @brief Test whether a boundary has a boundary type value
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_hasValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the boundary type value of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param value boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_getValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int *value);

/**
 * @brief Set the boundary type value of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param value boundary type value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_setValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int value);

/**
 * @brief Test whether a boundary has a velocity
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_hasVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the velocity of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param velocity boundary velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_getVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float **velocity);

/**
 * @brief Set the velocity of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param velocity boundary velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_setVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float *velocity);

/**
 * @brief Test whether a boundary has a restore coefficient
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param has flag signifying whether a boundary has a restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_hasRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has);

/**
 * @brief Get the restore coefficient of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_getRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float *restore);

/**
 * @brief Set the restore coefficient of a boundary
 * 
 * @param handle populated handle
 * @param name name of boundary
 * @param restore restore coefficient
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCBoundaryConditionsArgsContainer_setRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float restore);

#endif // _WRAPS_C_MXCBOUNDARYCONDITIONS_H_