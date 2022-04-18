/**
 * @file MxCBond.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBond
 * @date 2022-04-01
 */

#include "MxCBond.h"

#include "mechanica_c_private.h"

#include <fptype.h>
#include <io/mx_io.h>
#include <bond.h>
#include <angle.h>
#include <dihedral.h>
#include <MxPotential.h>
#include <MxStyle.h>

namespace mx { 

MxBondHandle *castC(struct MxBondHandleHandle *handle) {
    return castC<MxBondHandle, MxBondHandleHandle>(handle);
}

MxAngleHandle *castC(struct MxAngleHandleHandle *handle) {
    return castC<MxAngleHandle, MxAngleHandleHandle>(handle);
}

MxDihedralHandle *castC(struct MxDihedralHandleHandle *handle) {
    return castC<MxDihedralHandle, MxDihedralHandleHandle>(handle);
}

}

#define MXBONDHANDLE_GET(handle, varname) \
    MxBondHandle *varname = mx::castC<MxBondHandle, MxBondHandleHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXANGLEHANDLE_GET(handle, varname) \
    MxAngleHandle *varname = mx::castC<MxAngleHandle, MxAngleHandleHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXDIHEDRALHANDLE_GET(handle, varname) \
    MxDihedralHandle *varname = mx::castC<MxDihedralHandle, MxDihedralHandleHandle>(handle); \
    MXCPTRCHECK(varname);


//////////////
// Generics //
//////////////


namespace mx { namespace capi {

template <typename O, typename H> 
HRESULT getBondId(H *handle, int *id) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(id);
    *id = bhandle->getId();
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondStr(H *handle, char **str, unsigned int *numChars) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    return mx::capi::str2Char(bhandle->str(), str, numChars);
}

template <typename O, typename H> 
HRESULT bondCheck(H *handle, bool *flag) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(flag);
    *flag = bhandle->check();
    return S_OK;
}

template <typename O, typename H> 
HRESULT bondDecays(H *handle, bool *flag) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(flag);
    *flag = bhandle->decays();
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondEnergy(H *handle, double *value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(value);
    *value = bhandle->getEnergy();
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondPotential(H *handle, struct MxPotentialHandle *potential) {
    MXCPTRCHECK(potential);
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MxPotential *_potential = bhandle->getPotential();
    MXCPTRCHECK(_potential);
    potential->MxObj = (void*)_potential;
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondDissociationEnergy(H *handle, float *value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(value);
    *value = bhandle->getDissociationEnergy();
    return S_OK;
}

template <typename O, typename H> 
HRESULT setBondDissociationEnergy(H *handle, const float &value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    bhandle->setDissociationEnergy(value);
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondHalfLife(H *handle, float *value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(value);
    *value = bhandle->getHalfLife();
    return S_OK;
}

template <typename O, typename H> 
HRESULT setBondHalfLife(H *handle, const float &value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    bhandle->setHalfLife(value);
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondActive(H *handle, bool *flag) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(flag);
    *flag = bhandle->getActive();
    return S_OK;
}

template <typename O, typename H, typename B> 
HRESULT setBondActive(H *handle, const bool &flag, const unsigned int &activeFlag) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    B *b = bhandle->get();
    b->flags |= activeFlag;
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondStyle(H *handle, struct MxStyleHandle *style) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(style);
    style->MxObj = (void*)bhandle->getStyle();
    return S_OK;
}

template <typename O, typename H> 
HRESULT setBondStyle(H *handle, struct MxStyleHandle *style) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    MXCPTRCHECK(style);
    MXCPTRCHECK(style->MxObj);
    bhandle->setStyle((MxStyle*)style->MxObj);
    return S_OK;
}

template <typename O, typename H> 
HRESULT getBondAge(H *handle, double *value) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle) 
    MXCPTRCHECK(value);
    *value = bhandle->getAge();
    return S_OK;
}

template <typename O, typename H, typename B> 
HRESULT bondToString(H *handle, char **str, unsigned int *numChars) {
    O *bhandle = castC<O, H>(handle);
    MXCPTRCHECK(bhandle);
    B *b = bhandle->get();
    MXCPTRCHECK(b);
    return mx::capi::str2Char(b->toString(), str, numChars);
}

template <typename O, typename H, typename B> 
HRESULT bondFromString(H *handle, const char *str) {
    MXCPTRCHECK(str);
    B *b = B::fromString(str);
    MXCPTRCHECK(b);
    O *bhandle = new O(b->id);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

template <typename O, typename H> 
HRESULT getAllBonds(H **handles, unsigned int *numBonds) {
    MXCPTRCHECK(handles);
    MXCPTRCHECK(numBonds);

    auto _items = O::items();
    *numBonds = _items.size();
    if(*numBonds == 0) 
        return S_OK;

    H *_handles = (H*)malloc(*numBonds * sizeof(H));
    if(!_handles) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < *numBonds; i++) 
        _handles[i].MxObj = (void*)_items[i];
    *handles = _handles;
    return S_OK;
}

HRESULT passBondIdsForParticle(const std::vector<int32_t> &items, unsigned int **bids, unsigned int *numIds) {
    MXCPTRCHECK(bids);
    MXCPTRCHECK(numIds);

    *numIds = items.size();
    if(*numIds == 0) 
        return S_OK;

    unsigned int *_bids = (unsigned int*)malloc(*numIds * sizeof(unsigned int));
    if(!_bids) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < *numIds; i++) 
        _bids[i] = items[i];
    *bids = _bids;
    return S_OK;
}

}}


//////////////////
// MxBondHandle //
//////////////////


HRESULT MxCBondHandle_init(struct MxBondHandleHandle *handle, unsigned int id) {
    MXCPTRCHECK(handle);
    MxBondHandle *bhandle = new MxBondHandle(id);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCBondHandle_create(struct MxBondHandleHandle *handle, 
                             struct MxPotentialHandle *potential,
                             struct MxParticleHandleHandle *i, 
                             struct MxParticleHandleHandle *j) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(potential); MXCPTRCHECK(potential->MxObj);
    MXCPTRCHECK(i); MXCPTRCHECK(i->MxObj);
    MXCPTRCHECK(j); MXCPTRCHECK(j->MxObj);
    MxBondHandle *bhandle = MxBond::create((MxPotential*)potential->MxObj, 
                                           (MxParticleHandle*)i->MxObj, 
                                           (MxParticleHandle*)j->MxObj);
    MXCPTRCHECK(bhandle);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCBondHandle_getId(struct MxBondHandleHandle *handle, int *id) {
    return mx::capi::getBondId<MxBondHandle, MxBondHandleHandle>(handle, id);
}

HRESULT MxCBondHandle_getStr(struct MxBondHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::getBondStr<MxBondHandle, MxBondHandleHandle>(handle, str, numChars);
}

HRESULT MxCBondHandle_check(struct MxBondHandleHandle *handle, bool *flag) {
    return mx::capi::bondCheck<MxBondHandle, MxBondHandleHandle>(handle, flag);
}

HRESULT MxCBondHandle_destroy(struct MxBondHandleHandle *handle) {
    return mx::capi::destroyHandle<MxBondHandle, MxBondHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCBondHandle_decays(struct MxBondHandleHandle *handle, bool *flag) {
    return mx::capi::bondDecays<MxBondHandle, MxBondHandleHandle>(handle, flag);
}

HRESULT MxCBondHandle_getEnergy(struct MxBondHandleHandle *handle, double *value) {
    return mx::capi::getBondEnergy<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_getParts(struct MxBondHandleHandle *handle, int *parti, int *partj) {
    MXBONDHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(parti);
    MXCPTRCHECK(partj);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    return S_OK;
}

HRESULT MxCBondHandle_getPotential(struct MxBondHandleHandle *handle, struct MxPotentialHandle *potential) {
    return mx::capi::getBondPotential<MxBondHandle, MxBondHandleHandle>(handle, potential);
}

HRESULT MxCBondHandle_getDissociationEnergy(struct MxBondHandleHandle *handle, float *value) {
    return mx::capi::getBondDissociationEnergy<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_setDissociationEnergy(struct MxBondHandleHandle *handle, float value) {
    return mx::capi::setBondDissociationEnergy<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_getHalfLife(struct MxBondHandleHandle *handle, float *value) {
    return mx::capi::getBondHalfLife<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_setHalfLife(struct MxBondHandleHandle *handle, float value) {
    return mx::capi::setBondHalfLife<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_getActive(struct MxBondHandleHandle *handle, bool *flag) {
    return mx::capi::getBondActive<MxBondHandle, MxBondHandleHandle>(handle, flag);
}

HRESULT MxCBondHandle_setActive(struct MxBondHandleHandle *handle, bool flag) {
    return mx::capi::setBondActive<MxBondHandle, MxBondHandleHandle, MxBond>(handle, flag, BOND_ACTIVE);
}

HRESULT MxCBondHandle_getStyle(struct MxBondHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::getBondStyle<MxBondHandle, MxBondHandleHandle>(handle, style);
}

HRESULT MxCBondHandle_setStyle(struct MxBondHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::setBondStyle<MxBondHandle, MxBondHandleHandle>(handle, style);
}

HRESULT MxCBondHandle_getAge(struct MxBondHandleHandle *handle, double *value) {
    return mx::capi::getBondAge<MxBondHandle, MxBondHandleHandle>(handle, value);
}

HRESULT MxCBondHandle_toString(struct MxBondHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::bondToString<MxBondHandle, MxBondHandleHandle, MxBond>(handle, str, numChars);
}

HRESULT MxCBondHandle_fromString(struct MxBondHandleHandle *handle, const char *str) {
    return mx::capi::bondFromString<MxBondHandle, MxBondHandleHandle, MxBond>(handle, str);
}


///////////////////
// MxAngleHandle //
///////////////////


HRESULT MxCAngleHandle_init(struct MxAngleHandleHandle *handle, unsigned int id) {
    MXCPTRCHECK(handle);
    MxAngleHandle *bhandle = new MxAngleHandle(id);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCAngleHandle_create(struct MxAngleHandleHandle *handle, 
                              struct MxPotentialHandle *potential,
                              struct MxParticleHandleHandle *i, 
                              struct MxParticleHandleHandle *j, 
                              struct MxParticleHandleHandle *k) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(potential); MXCPTRCHECK(potential->MxObj);
    MXCPTRCHECK(i); MXCPTRCHECK(i->MxObj);
    MXCPTRCHECK(j); MXCPTRCHECK(j->MxObj);
    MxAngleHandle *bhandle = MxAngle::create((MxPotential*)potential->MxObj, 
                                             (MxParticleHandle*)i->MxObj, 
                                             (MxParticleHandle*)j->MxObj, 
                                             (MxParticleHandle*)k->MxObj);
    MXCPTRCHECK(bhandle);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCAngleHandle_getId(struct MxAngleHandleHandle *handle, int *id) {
    return mx::capi::getBondId<MxAngleHandle, MxAngleHandleHandle>(handle, id);
}

HRESULT MxCAngleHandle_getStr(struct MxAngleHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::getBondStr<MxAngleHandle, MxAngleHandleHandle>(handle, str, numChars);
}

HRESULT MxCAngleHandle_check(struct MxAngleHandleHandle *handle, bool *flag) {
    return mx::capi::bondCheck<MxAngleHandle, MxAngleHandleHandle>(handle, flag);
}

HRESULT MxCAngleHandle_destroy(struct MxAngleHandleHandle *handle) {
    return mx::capi::destroyHandle<MxAngleHandle, MxAngleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCAngleHandle_decays(struct MxAngleHandleHandle *handle, bool *flag) {
    return mx::capi::bondDecays<MxAngleHandle, MxAngleHandleHandle>(handle, flag);
}

HRESULT MxCAngleHandle_getEnergy(struct MxAngleHandleHandle *handle, double *value) {
    return mx::capi::getBondEnergy<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_getParts(struct MxAngleHandleHandle *handle, int *parti, int *partj, int *partk) {
    MXANGLEHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(parti);
    MXCPTRCHECK(partj);
    MXCPTRCHECK(partk);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    *partk = pids[2];
    return S_OK;
}

HRESULT MxCAngleHandle_getPotential(struct MxAngleHandleHandle *handle, struct MxPotentialHandle *potential) {
    return mx::capi::getBondPotential<MxAngleHandle, MxAngleHandleHandle>(handle, potential);
}

HRESULT MxCAngleHandle_getDissociationEnergy(struct MxAngleHandleHandle *handle, float *value) {
    return mx::capi::getBondDissociationEnergy<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_setDissociationEnergy(struct MxAngleHandleHandle *handle, float value) {
    return mx::capi::setBondDissociationEnergy<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_getHalfLife(struct MxAngleHandleHandle *handle, float *value) {
    return mx::capi::getBondHalfLife<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_setHalfLife(struct MxAngleHandleHandle *handle, float value) {
    return mx::capi::setBondHalfLife<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_getActive(struct MxAngleHandleHandle *handle, bool *flag) {
    return mx::capi::getBondActive<MxAngleHandle, MxAngleHandleHandle>(handle, flag);
}

HRESULT MxCAngleHandle_setActive(struct MxAngleHandleHandle *handle, bool flag) {
    return mx::capi::setBondActive<MxAngleHandle, MxAngleHandleHandle, MxAngle>(handle, flag, ANGLE_ACTIVE);
}

HRESULT MxCAngleHandle_getStyle(struct MxAngleHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::getBondStyle<MxAngleHandle, MxAngleHandleHandle>(handle, style);
}

HRESULT MxCAngleHandle_setStyle(struct MxAngleHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::setBondStyle<MxAngleHandle, MxAngleHandleHandle>(handle, style);
}

HRESULT MxCAngleHandle_getAge(struct MxAngleHandleHandle *handle, double *value) {
    return mx::capi::getBondAge<MxAngleHandle, MxAngleHandleHandle>(handle, value);
}

HRESULT MxCAngleHandle_toString(struct MxAngleHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::bondToString<MxAngleHandle, MxAngleHandleHandle, MxAngle>(handle, str, numChars);
}

HRESULT MxCAngleHandle_fromString(struct MxAngleHandleHandle *handle, const char *str) {
    return mx::capi::bondFromString<MxAngleHandle, MxAngleHandleHandle, MxAngle>(handle, str);
}


//////////////////////
// MxDihedralHandle //
//////////////////////


HRESULT MxCDihedralHandle_init(struct MxDihedralHandleHandle *handle, unsigned int id) {
    MXCPTRCHECK(handle);
    MxDihedralHandle *bhandle = new MxDihedralHandle(id);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCDihedralHandle_create(struct MxDihedralHandleHandle *handle, 
                                 struct MxPotentialHandle *potential,
                                 struct MxParticleHandleHandle *i, 
                                 struct MxParticleHandleHandle *j, 
                                 struct MxParticleHandleHandle *k, 
                                 struct MxParticleHandleHandle *l) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(potential); MXCPTRCHECK(potential->MxObj);
    MXCPTRCHECK(i); MXCPTRCHECK(i->MxObj);
    MXCPTRCHECK(j); MXCPTRCHECK(j->MxObj);
    MXCPTRCHECK(k); MXCPTRCHECK(k->MxObj);
    MXCPTRCHECK(l); MXCPTRCHECK(l->MxObj);
    MxDihedralHandle *bhandle = MxDihedral::create((MxPotential*)potential->MxObj, 
                                                   (MxParticleHandle*)i->MxObj, 
                                                   (MxParticleHandle*)j->MxObj, 
                                                   (MxParticleHandle*)k->MxObj, 
                                                   (MxParticleHandle*)l->MxObj);
    MXCPTRCHECK(bhandle);
    handle->MxObj = (void*)bhandle;
    return S_OK;
}

HRESULT MxCDihedralHandle_getId(struct MxDihedralHandleHandle *handle, int *id) {
    return mx::capi::getBondId<MxDihedralHandle, MxDihedralHandleHandle>(handle, id);
}

HRESULT MxCDihedralHandle_getStr(struct MxDihedralHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::getBondStr<MxDihedralHandle, MxDihedralHandleHandle>(handle, str, numChars);
}

HRESULT MxCDihedralHandle_check(struct MxDihedralHandleHandle *handle, bool *flag) {
    return mx::capi::bondCheck<MxDihedralHandle, MxDihedralHandleHandle>(handle, flag);
}

HRESULT MxCDihedralHandle_destroy(struct MxDihedralHandleHandle *handle) {
    return mx::capi::destroyHandle<MxDihedralHandle, MxDihedralHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCDihedralHandle_decays(struct MxDihedralHandleHandle *handle, bool *flag) {
    return mx::capi::bondDecays<MxDihedralHandle, MxDihedralHandleHandle>(handle, flag);
}

HRESULT MxCDihedralHandle_getEnergy(struct MxDihedralHandleHandle *handle, double *value) {
    return mx::capi::getBondEnergy<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_getParts(struct MxDihedralHandleHandle *handle, int *parti, int *partj, int *partk, int *partl) {
    MXDIHEDRALHANDLE_GET(handle, bhandle);
    MXCPTRCHECK(parti);
    MXCPTRCHECK(partj);
    MXCPTRCHECK(partk);
    MXCPTRCHECK(partl);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    *partk = pids[2];
    *partl = pids[3];
    return S_OK;
}

HRESULT MxCDihedralHandle_getPotential(struct MxDihedralHandleHandle *handle, struct MxPotentialHandle *potential) {
    return mx::capi::getBondPotential<MxDihedralHandle, MxDihedralHandleHandle>(handle, potential);
}

HRESULT MxCDihedralHandle_getDissociationEnergy(struct MxDihedralHandleHandle *handle, float *value) {
    return mx::capi::getBondDissociationEnergy<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_setDissociationEnergy(struct MxDihedralHandleHandle *handle, float value) {
    return mx::capi::setBondDissociationEnergy<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_getHalfLife(struct MxDihedralHandleHandle *handle, float *value) {
    return mx::capi::getBondHalfLife<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_setHalfLife(struct MxDihedralHandleHandle *handle, float value) {
    return mx::capi::setBondHalfLife<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_getActive(struct MxDihedralHandleHandle *handle, bool *flag) {
    return mx::capi::getBondActive<MxDihedralHandle, MxDihedralHandleHandle>(handle, flag);
}

HRESULT MxCDihedralHandle_setActive(struct MxDihedralHandleHandle *handle, bool flag) {
    return mx::capi::setBondActive<MxDihedralHandle, MxDihedralHandleHandle, MxDihedral>(handle, flag, DIHEDRAL_ACTIVE);
}

HRESULT MxCDihedralHandle_getStyle(struct MxDihedralHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::getBondStyle<MxDihedralHandle, MxDihedralHandleHandle>(handle, style);
}

HRESULT MxCDihedralHandle_setStyle(struct MxDihedralHandleHandle *handle, struct MxStyleHandle *style) {
    return mx::capi::setBondStyle<MxDihedralHandle, MxDihedralHandleHandle>(handle, style);
}

HRESULT MxCDihedralHandle_getAge(struct MxDihedralHandleHandle *handle, double *value) {
    return mx::capi::getBondAge<MxDihedralHandle, MxDihedralHandleHandle>(handle, value);
}

HRESULT MxCDihedralHandle_toString(struct MxDihedralHandleHandle *handle, char **str, unsigned int *numChars) {
    return mx::capi::bondToString<MxDihedralHandle, MxDihedralHandleHandle, MxDihedral>(handle, str, numChars);
}

HRESULT MxCDihedralHandle_fromString(struct MxDihedralHandleHandle *handle, const char *str) {
    return mx::capi::bondFromString<MxDihedralHandle, MxDihedralHandleHandle, MxDihedral>(handle, str);
}

//////////////////////
// Module functions //
//////////////////////


HRESULT MxCBondHandle_getAll(struct MxBondHandleHandle **handles, unsigned int *numBonds) {
    return mx::capi::getAllBonds<MxBondHandle, MxBondHandleHandle>(handles, numBonds);
}

HRESULT MxCBond_pairwise(struct MxPotentialHandle *pot, 
                         struct MxParticleListHandle *parts, 
                         double cutoff, 
                         struct MxParticleTypeHandle *ppairsA, 
                         struct MxParticleTypeHandle *ppairsB, 
                         unsigned int numTypePairs, 
                         double *half_life, 
                         double *bond_energy, 
                         struct MxBondHandleHandle **bonds, 
                         unsigned int *numBonds) 
{
    MXCPTRCHECK(pot); MXCPTRCHECK(pot->MxObj);
    MXCPTRCHECK(parts); MXCPTRCHECK(parts->MxObj);
    MXCPTRCHECK(ppairsA);
    MXCPTRCHECK(ppairsB);
    MXCPTRCHECK(bonds);

    std::vector<std::pair<MxParticleType*, MxParticleType*> *> ppairs;
    MxParticleTypeHandle pta, ptb;
    for(unsigned int i = 0; i < numTypePairs; i++) {
        pta = ppairsA[i];
        ptb = ppairsB[i];
        if(!pta.MxObj || !ptb.MxObj) 
            return E_FAIL;
        ppairs.push_back(new std::pair<MxParticleType*, MxParticleType*>(std::make_pair((MxParticleType*)pta.MxObj, (MxParticleType*)ptb.MxObj)));
    }
    double _half_life = half_life ? *half_life : 0.0;
    double _bond_energy = bond_energy ? *bond_energy : 0.0;
    auto _items = MxBondHandle::pairwise((MxPotential*)pot->MxObj, (MxParticleList*)parts->MxObj, cutoff, &ppairs, _half_life, _bond_energy, BOND_ACTIVE);
    for(unsigned int i = 0; i < numTypePairs; i++) 
        delete ppairs[i];
    if(!_items)
        return E_FAIL;

    *numBonds = _items->size();
    if(*numBonds == 0) 
        return S_OK;

    MxBondHandleHandle *_bonds = (MxBondHandleHandle*)malloc(*numBonds * sizeof(MxBondHandleHandle));
    if(!_bonds) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < *numBonds; i++) 
        _bonds[i].MxObj = (void*)(*_items)[i];
    *bonds = _bonds;
    return S_OK;
}

HRESULT MxCBond_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    MXCPTRCHECK(bids)
    return mx::capi::passBondIdsForParticle(MxBond_IdsForParticle(pid), bids, numIds);
}

HRESULT MxCBond_destroyAll() {
    return MxBond_DestroyAll();
}

HRESULT MxCAngleHandle_getAll(struct MxAngleHandleHandle **handles, unsigned int *numBonds) {
    return mx::capi::getAllBonds<MxAngleHandle, MxAngleHandleHandle>(handles, numBonds);
}

HRESULT MxCAngle_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    MXCPTRCHECK(bids);
    return mx::capi::passBondIdsForParticle(MxAngle_IdsForParticle(pid), bids, numIds);
}

HRESULT MxCAngle_destroyAll() {
    return MxAngle_DestroyAll();
}

HRESULT MxCDihedralHandle_getAll(struct MxDihedralHandleHandle **handles, unsigned int *numBonds) {
    return mx::capi::getAllBonds<MxDihedralHandle, MxDihedralHandleHandle>(handles, numBonds);
}

HRESULT MxCDihedral_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    MXCPTRCHECK(bids);
    return mx::capi::passBondIdsForParticle(MxDihedral_IdsForParticle(pid), bids, numIds);
}

HRESULT MxCDihedral_destroyAll() {
    return MxDihedral_DestroyAll();
}
