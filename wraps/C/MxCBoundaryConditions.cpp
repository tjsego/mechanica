/**
 * @file MxCBoundaryConditions.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBoundaryConditions
 * @date 2022-03-31
 */

#include "MxCBoundaryConditions.h"

#include "mechanica_c_private.h"

#include <MxBoundaryConditions.hpp>
#include <space.h>


namespace mx { 

MxBoundaryCondition *castC(struct MxBoundaryConditionHandle *handle) {
    return castC<MxBoundaryCondition, MxBoundaryConditionHandle>(handle);
}

MxBoundaryConditions *castC(struct MxBoundaryConditionsHandle *handle) {
    return castC<MxBoundaryConditions, MxBoundaryConditionsHandle>(handle);
}

MxBoundaryConditionsArgsContainer *castC(struct MxBoundaryConditionsArgsContainerHandle *handle) {
    return castC<MxBoundaryConditionsArgsContainer, MxBoundaryConditionsArgsContainerHandle>(handle);
}

}

#define MXBOUNDARYCONDITIONHANDLE_GET(handle, varname) \
    MxBoundaryCondition *varname = mx::castC<MxBoundaryCondition, MxBoundaryConditionHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXBOUNDARYCONDITIONSHANDLE_GET(handle, varname) \
    MxBoundaryConditions *varname = mx::castC<MxBoundaryConditions, MxBoundaryConditionsHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, varname) \
    MxBoundaryConditionsArgsContainer *varname = mx::castC<MxBoundaryConditionsArgsContainer, MxBoundaryConditionsArgsContainerHandle>(handle); \
    MXCPTRCHECK(varname);


////////////////////////////////
// BoundaryConditionSpaceKind //
////////////////////////////////


HRESULT MxCBoundaryConditionSpaceKind_init(struct BoundaryConditionSpaceKindHandle *handle) {
    handle->SPACE_PERIODIC_X = space_periodic_x;
    handle->SPACE_PERIODIC_Y = space_periodic_y;
    handle->SPACE_PERIODIC_Z = space_periodic_z;
    handle->SPACE_PERIODIC_FULL = space_periodic_full;
    handle->SPACE_FREESLIP_X = SPACE_FREESLIP_X;
    handle->SPACE_FREESLIP_Y = SPACE_FREESLIP_Y;
    handle->SPACE_FREESLIP_Z = SPACE_FREESLIP_Z;
    handle->SPACE_FREESLIP_FULL = SPACE_FREESLIP_FULL;
    return S_OK;
}


///////////////////////////
// BoundaryConditionKind //
///////////////////////////


HRESULT MxCBoundaryConditionKind_init(struct BoundaryConditionKindHandle *handle) {
    handle->BOUNDARY_PERIODIC = BOUNDARY_PERIODIC;
    handle->BOUNDARY_FREESLIP = BOUNDARY_FREESLIP;
    handle->BOUNDARY_RESETTING = BOUNDARY_RESETTING;
    return S_OK;
}


/////////////////////////
// MxBoundaryCondition //
/////////////////////////


HRESULT MxCBoundaryCondition_getId(struct MxBoundaryConditionHandle *handle, int *bid) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bid);
    *bid = bhandle->id;
    return S_OK;
}

HRESULT MxCBoundaryCondition_getVelocity(struct MxBoundaryConditionHandle *handle, float **velocity) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(velocity);
    MXVECTOR3_COPYFROM(bhandle->velocity, (*velocity));
    return S_OK;
}

HRESULT MxCBoundaryCondition_setVelocity(struct MxBoundaryConditionHandle *handle, float *velocity) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(velocity);
    MXVECTOR3_COPYTO(velocity, bhandle->velocity);
    return S_OK;
}

HRESULT MxCBoundaryCondition_getRestore(struct MxBoundaryConditionHandle *handle, float *restore) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(restore);
    *restore = bhandle->restore;
    return S_OK;
}

HRESULT MxCBoundaryCondition_setRestore(struct MxBoundaryConditionHandle *handle, float restore) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    bhandle->restore = restore;
    return S_OK;
}

HRESULT MxCBoundaryCondition_getNormal(struct MxBoundaryConditionHandle *handle, float **normal) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(normal);
    MXVECTOR3_COPYFROM(bhandle->normal, (*normal));
    return S_OK;
}

HRESULT MxCBoundaryCondition_getRadius(struct MxBoundaryConditionHandle *handle, float *radius) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(radius);
    *radius = bhandle->radius;
    return S_OK;
}

HRESULT MxCBoundaryCondition_setRadius(struct MxBoundaryConditionHandle *handle, float radius) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    bhandle->radius = radius;
    return S_OK;
}

HRESULT MxCBoundaryCondition_setPotential(struct MxBoundaryConditionHandle *handle, struct MxParticleTypeHandle *partHandle, struct MxPotentialHandle *potHandle) {
    MXBOUNDARYCONDITIONHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(partHandle); MXCPTRCHECK(partHandle->MxObj);
    MXCPTRCHECK(potHandle); MXCPTRCHECK(potHandle->MxObj);
    bhandle->set_potential((MxParticleType*)partHandle->MxObj, (MxPotential*)potHandle->MxObj);
    return S_OK;
}


//////////////////////////
// MxBoundaryConditions //
//////////////////////////


HRESULT MxCBoundaryConditions_getTop(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->top;
    return S_OK;
}

HRESULT MxCBoundaryConditions_getBottom(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->bottom;
    return S_OK;
}

HRESULT MxCBoundaryConditions_getLeft(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->left;
    return S_OK;
}

HRESULT MxCBoundaryConditions_getRight(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->right;
    return S_OK;
}

HRESULT MxCBoundaryConditions_getFront(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->front;
    return S_OK;
}

HRESULT MxCBoundaryConditions_getBack(struct MxBoundaryConditionsHandle *handle, struct MxBoundaryConditionHandle *bchandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(bchandle);
    bchandle->MxObj = (void*)&bhandle->back;
    return S_OK;
}

HRESULT MxCBoundaryConditions_setPotential(struct MxBoundaryConditionsHandle *handle, struct MxParticleTypeHandle *partHandle, struct MxPotentialHandle *potHandle) {
    MXBOUNDARYCONDITIONSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(partHandle); MXCPTRCHECK(partHandle->MxObj);
    MXCPTRCHECK(potHandle); MXCPTRCHECK(potHandle->MxObj);
    bhandle->set_potential((MxParticleType*)partHandle->MxObj, (MxPotential*)potHandle->MxObj);
    return S_OK;
}


///////////////////////////////////////
// MxBoundaryConditionsArgsContainer //
///////////////////////////////////////


HRESULT MxCBoundaryConditionsArgsContainer_init(struct MxBoundaryConditionsArgsContainerHandle *handle) {
    MXCPTRCHECK(handle);
    MxBoundaryConditionsArgsContainer *bargs = new MxBoundaryConditionsArgsContainer();
    handle->MxObj = (void*)bargs;
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_destroy(struct MxBoundaryConditionsArgsContainerHandle *handle) {
    return mx::capi::destroyHandle<MxBoundaryConditionsArgsContainer, MxBoundaryConditionsArgsContainerHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCBoundaryConditionsArgsContainer_hasValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, bool *has) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(has);
    *has = bhandle->bcValue != NULL;
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_getValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, unsigned int *_bcValue) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(_bcValue);
    MXCPTRCHECK(bhandle->bcValue);
    *_bcValue = *bhandle->bcValue;
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_setValueAll(struct MxBoundaryConditionsArgsContainerHandle *handle, unsigned int _bcValue) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    bhandle->setValueAll(_bcValue);
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_hasValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(has);
    *has = bhandle->bcVals == NULL ? false : bhandle->bcVals->find(name) != bhandle->bcVals->end();
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_getValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int *value) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(value);
    MXCPTRCHECK(bhandle->bcVals);
    auto itr = bhandle->bcVals->find(name);
    if(itr == bhandle->bcVals->end()) 
        return E_FAIL;
    *value = itr->second;
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_setValue(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, unsigned int value) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    bhandle->setValue(name, value);
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_hasVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(has);
    *has = bhandle->bcVels == NULL ? false : bhandle->bcVels->find(name) != bhandle->bcVels->end();
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_getVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float **velocity) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(velocity);
    MXCPTRCHECK(bhandle->bcVels);
    auto itr = bhandle->bcVels->find(name);
    if(itr == bhandle->bcVels->end()) 
        return E_FAIL;
    MxVector3f _velocity = itr->second;
    MXVECTOR3_COPYFROM(_velocity, (*velocity));
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_setVelocity(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float *velocity) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(velocity);
    bhandle->setVelocity(name, MxVector3f::from(velocity));
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_hasRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, bool *has) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(has);
    *has = bhandle->bcRestores == NULL ? false : bhandle->bcRestores->find(name) != bhandle->bcRestores->end();
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_getRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float *restore) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    MXCPTRCHECK(restore);
    MXCPTRCHECK(bhandle->bcRestores);
    auto itr = bhandle->bcRestores->find(name);
    if(itr == bhandle->bcRestores->end()) 
        return E_FAIL;
    *restore = itr->second;
    return S_OK;
}

HRESULT MxCBoundaryConditionsArgsContainer_setRestore(struct MxBoundaryConditionsArgsContainerHandle *handle, const char *name, float restore) {
    MXBOUNDARYCONDITIONSARGSHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(name);
    bhandle->setRestore(name, restore);
    return S_OK;
}
