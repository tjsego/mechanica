/**
 * @file MxCcell_polarity.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for center model cell polarity
 * @date 2022-04-07
 */

#include "MxCcell_polarity.h"

#include "../../../mechanica_c_private.h"

#include <models/center/cell_polarity/cell_polarity.h>


//////////////////
// Module casts //
//////////////////


namespace mx { 

PolarityForcePersistent *castC(struct PolarityForcePersistentHandle *handle) {
    return castC<PolarityForcePersistent, PolarityForcePersistentHandle>(handle);
}

MxPolarityArrowData *castC(struct MxPolarityArrowDataHandle *handle) {
    return castC<MxPolarityArrowData, MxPolarityArrowDataHandle>(handle);
}

MxCellPolarityPotentialContact *castC(struct MxCellPolarityPotentialContactHandle *handle) {
    return castC<MxCellPolarityPotentialContact, MxCellPolarityPotentialContactHandle>(handle);
}

}

#define POLARITYFORCEPERSISTENTHANDLE_GET(handle, varname) \
    PolarityForcePersistent *varname = mx::castC<PolarityForcePersistent, PolarityForcePersistentHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXPOLARITYARROWDATAHANDLE_GET(handle, varname) \
    MxPolarityArrowData *varname = mx::castC<MxPolarityArrowData, MxPolarityArrowDataHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, varname) \
    MxCellPolarityPotentialContact *varname = mx::castC<MxCellPolarityPotentialContact, MxCellPolarityPotentialContactHandle>(handle); \
    MXCPTRCHECK(varname);


//////////////////////
// PolarContactType //
//////////////////////


HRESULT MxCPolarContactType_init(struct PolarContactTypeEnumHandle *handle) {
    MXCPTRCHECK(handle);
    handle->REGULAR = PolarContactType::REGULAR;
    handle->ISOTROPIC = PolarContactType::ISOTROPIC;
    handle->ANISOTROPIC = PolarContactType::ANISOTROPIC;
    return S_OK;
}


/////////////////////////////
// PolarityForcePersistent //
/////////////////////////////


HRESULT MxCPolarityForcePersistent_init(struct PolarityForcePersistentHandle *handle, float sensAB, float sensPCP) {
    MXCPTRCHECK(handle);
    PolarityForcePersistent *f = new PolarityForcePersistent();
    f->sensAB = sensAB;
    f->sensPCP = sensPCP;
    handle->MxObj = (void*)f;
    return S_OK;
}

HRESULT MxCPolarityForcePersistent_destroy(struct PolarityForcePersistentHandle *handle) {
    return mx::capi::destroyHandle<PolarityForcePersistent, struct PolarityForcePersistentHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCPolarityForcePersistent_toBase(struct PolarityForcePersistentHandle *handle, struct MxForceHandle *baseHandle) {
    POLARITYFORCEPERSISTENTHANDLE_GET(handle, pf);
    MXCPTRCHECK(baseHandle);
    MxForce *_pf = (MxForce*)pf;
    baseHandle->MxObj = (void*)_pf;
    return S_OK;
}

HRESULT MxCPolarityForcePersistent_fromBase(struct MxForceHandle *baseHandle, struct PolarityForcePersistentHandle *handle) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(baseHandle); MXCPTRCHECK(baseHandle->MxObj);
    PolarityForcePersistent *pf = (PolarityForcePersistent*)baseHandle->MxObj;
    handle->MxObj = (void*)pf;
    return S_OK;
}


////////////////////////////////////
// MxCellPolarityPotentialContact //
////////////////////////////////////


const char *cTypeMap[] = {
    "regular", 
    "isotropic", 
    "anisotropic"
};


HRESULT MxCCellPolarityPotentialContact_init(struct MxCellPolarityPotentialContactHandle *handle, 
                                             float cutoff, 
                                             float couplingFlat, 
                                             float couplingOrtho, 
                                             float couplingLateral, 
                                             float distanceCoeff, 
                                             unsigned int cType, 
                                             float mag, 
                                             float rate, 
                                             float bendingCoeff) 
{
    MXCPTRCHECK(handle);
    MxCellPolarityPotentialContact *pc = mx::models::center::CellPolarity::potentialContact(
        cutoff, mag, rate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, cTypeMap[cType], bendingCoeff
    );
    handle->MxObj = (void*)pc;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_destroy(struct MxCellPolarityPotentialContactHandle *handle) {
    return mx::capi::destroyHandle<MxCellPolarityPotentialContact, MxCellPolarityPotentialContactHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCCellPolarityPotentialContact_toBase(struct MxCellPolarityPotentialContactHandle *handle, struct MxPotentialHandle *baseHandle) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(baseHandle);
    MxPotential *_pc = (MxPotential*)pc;
    baseHandle->MxObj = (void*)_pc;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_fromBase(struct MxPotentialHandle *baseHandle, struct MxCellPolarityPotentialContactHandle *handle) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(baseHandle); MXCPTRCHECK(baseHandle->MxObj);
    MxPotential *_pc = (MxPotential*)baseHandle->MxObj;
    MxCellPolarityPotentialContact *pc = (MxCellPolarityPotentialContact*)_pc;
    handle->MxObj = (void*)pc;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getCouplingFlat(struct MxCellPolarityPotentialContactHandle *handle, float *couplingFlat) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(couplingFlat);
    *couplingFlat = pc->couplingFlat;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setCouplingFlat(struct MxCellPolarityPotentialContactHandle *handle, float couplingFlat) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->couplingFlat = couplingFlat;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getCouplingOrtho(struct MxCellPolarityPotentialContactHandle *handle, float *couplingOrtho) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(couplingOrtho);
    *couplingOrtho = pc->couplingOrtho;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setCouplingOrtho(struct MxCellPolarityPotentialContactHandle *handle, float couplingOrtho) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->couplingOrtho = couplingOrtho;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getCouplingLateral(struct MxCellPolarityPotentialContactHandle *handle, float *couplingLateral) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(couplingLateral);
    *couplingLateral = pc->couplingLateral;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setCouplingLateral(struct MxCellPolarityPotentialContactHandle *handle, float couplingLateral) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->couplingLateral = couplingLateral;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getDistanceCoeff(struct MxCellPolarityPotentialContactHandle *handle, float *distanceCoeff) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(distanceCoeff);
    *distanceCoeff = pc->distanceCoeff;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setDistanceCoeff(struct MxCellPolarityPotentialContactHandle *handle, float distanceCoeff) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->distanceCoeff = distanceCoeff;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getCType(struct MxCellPolarityPotentialContactHandle *handle, unsigned int *cType) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(cType);
    *cType = (unsigned int)pc->cType;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setCType(struct MxCellPolarityPotentialContactHandle *handle, unsigned int cType) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->cType = (PolarContactType)cType;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getMag(struct MxCellPolarityPotentialContactHandle *handle, float *mag) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(mag);
    *mag = pc->mag;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setMag(struct MxCellPolarityPotentialContactHandle *handle, float mag) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->mag = mag;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getRate(struct MxCellPolarityPotentialContactHandle *handle, float *rate) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(rate);
    *rate = pc->rate;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setRate(struct MxCellPolarityPotentialContactHandle *handle, float rate) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->rate = rate;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_getBendingCoeff(struct MxCellPolarityPotentialContactHandle *handle, float *bendingCoeff) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    MXCPTRCHECK(bendingCoeff);
    *bendingCoeff = pc->bendingCoeff;
    return S_OK;
}

HRESULT MxCCellPolarityPotentialContact_setBendingCoeff(struct MxCellPolarityPotentialContactHandle *handle, float bendingCoeff) {
    MXCELLPOLARITYPOTENTIALCONTACTHANDLE_GET(handle, pc);
    pc->bendingCoeff = bendingCoeff;
    return S_OK;
}


/////////////////////////
// MxPolarityArrowData //
/////////////////////////


HRESULT MxCPolarityArrowData_getArrowLength(struct MxPolarityArrowDataHandle *handle, float *arrowLength) {
    MXPOLARITYARROWDATAHANDLE_GET(handle, ad);
    MXCPTRCHECK(arrowLength);
    *arrowLength = ad->arrowLength;
    return S_OK;
}

HRESULT MxCPolarityArrowData_setArrowLength(struct MxPolarityArrowDataHandle *handle, float arrowLength) {
    MXPOLARITYARROWDATAHANDLE_GET(handle, ad);
    ad->arrowLength = arrowLength;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCCellPolarity_GetVectorAB(int pId, bool current, float **vec) {
    MXCPTRCHECK(vec);
    auto _vec = mx::models::center::CellPolarity::getVectorAB(pId, current);
    MXVECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT MxCCellPolarity_GetVectorPCP(int pId, bool current, float **vec) {
    MXCPTRCHECK(vec);
    auto _vec = mx::models::center::CellPolarity::getVectorPCP(pId, current);
    MXVECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT MxCCellPolarity_SetVectorAB(int pId, float *pVec, bool current, bool init) {
    MXCPTRCHECK(pVec);
    mx::models::center::CellPolarity::setVectorAB(pId, MxVector3f::from(pVec), current, init);
    return S_OK;
}

HRESULT MxCCellPolarity_SetVectorPCP(int pId, float *pVec, bool current, bool init) {
    MXCPTRCHECK(pVec);
    mx::models::center::CellPolarity::setVectorPCP(pId, MxVector3f::from(pVec), current, init);
    return S_OK;
}

HRESULT MxCCellPolarity_update() {
    mx::models::center::CellPolarity::update();
    return S_OK;
}

HRESULT MxCCellPolarity_registerParticle(struct MxParticleHandleHandle *ph) {
    MXCPTRCHECK(ph); MXCPTRCHECK(ph->MxObj);
    mx::models::center::CellPolarity::registerParticle((MxParticleHandle*)ph->MxObj);
    return S_OK;
}

HRESULT MxCCellPolarity_unregister(struct MxParticleHandleHandle *ph) {
    MXCPTRCHECK(ph); MXCPTRCHECK(ph->MxObj);
    mx::models::center::CellPolarity::unregister((MxParticleHandle*)ph->MxObj);
    return S_OK;
}

HRESULT MxCCellPolarity_registerType(struct MxParticleTypeHandle *pType, const char *initMode, float *initPolarAB, float *initPolarPCP) {
    MXCPTRCHECK(pType) MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(initMode);
    MXCPTRCHECK(initPolarAB);
    MXCPTRCHECK(initPolarPCP);
    mx::models::center::CellPolarity::registerType((MxParticleType*)pType->MxObj, initMode, MxVector3f::from(initPolarAB), MxVector3f::from(initPolarPCP));
    return S_OK;
}

HRESULT MxCCellPolarity_GetInitMode(struct MxParticleTypeHandle *pType, char **initMode, unsigned int *numChars) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    return mx::capi::str2Char(mx::models::center::CellPolarity::getInitMode((MxParticleType*)pType->MxObj), initMode, numChars);
}

HRESULT MxCCellPolarity_SetInitMode(struct MxParticleTypeHandle *pType, const char *value) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(value);
    mx::models::center::CellPolarity::setInitMode((MxParticleType*)pType->MxObj, value);
    return S_OK;
}

HRESULT MxCCellPolarity_GetInitPolarAB(struct MxParticleTypeHandle *pType, float **vec) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(vec);
    auto _vec = mx::models::center::CellPolarity::getInitPolarAB((MxParticleType*)pType->MxObj);
    MXVECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT MxCCellPolarity_SetInitPolarAB(struct MxParticleTypeHandle *pType, float *value) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(value);
    mx::models::center::CellPolarity::setInitPolarAB((MxParticleType*)pType->MxObj, MxVector3f::from(value));
    return S_OK;
}

HRESULT MxCCellPolarity_GetInitPolarPCP(struct MxParticleTypeHandle *pType, float **vec) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(vec);
    auto _vec = mx::models::center::CellPolarity::getInitPolarPCP((MxParticleType*)pType->MxObj);
    MXVECTOR3_COPYFROM(_vec, (*vec));
    return S_OK;
}

HRESULT MxCCellPolarity_SetInitPolarPCP(struct MxParticleTypeHandle *pType, float *value) {
    MXCPTRCHECK(pType); MXCPTRCHECK(pType->MxObj);
    MXCPTRCHECK(value);
    mx::models::center::CellPolarity::setInitPolarPCP((MxParticleType*)pType->MxObj, MxVector3f::from(value));
    return S_OK;
}

HRESULT MxCCellPolarity_SetDrawVectors(bool _draw) {
    mx::models::center::CellPolarity::setDrawVectors(_draw);
    return S_OK;
}

HRESULT MxCCellPolarity_SetArrowColors(const char *colorAB, const char *colorPCP) {
    MXCPTRCHECK(colorAB);
    MXCPTRCHECK(colorPCP);
    mx::models::center::CellPolarity::setArrowColors(colorAB, colorPCP);
    return S_OK;
}

HRESULT MxCCellPolarity_SetArrowScale(float _scale) {
    mx::models::center::CellPolarity::setArrowScale(_scale);
    return S_OK;
}

HRESULT MxCCellPolarity_SetArrowLength(float _length) {
    mx::models::center::CellPolarity::setArrowLength(_length);
    return S_OK;
}

HRESULT MxCCellPolarity_GetVectorArrowAB(unsigned int pId, struct MxPolarityArrowDataHandle *arrowData) {
    MXCPTRCHECK(arrowData);
    auto _arrowData = mx::models::center::CellPolarity::getVectorArrowAB(pId);
    MXCPTRCHECK(_arrowData);
    arrowData->MxObj = (void*)_arrowData;
    return S_OK;
}

HRESULT MxCCellPolarity_GetVectorArrowPCP(unsigned int pId, struct MxPolarityArrowDataHandle *arrowData) {
    MXCPTRCHECK(arrowData);
    auto _arrowData = mx::models::center::CellPolarity::getVectorArrowPCP(pId);
    MXCPTRCHECK(_arrowData);
    arrowData->MxObj = (void*)_arrowData;
    return S_OK;
}

HRESULT MxCCellPolarity_load() {
    mx::models::center::CellPolarity::load();
    return S_OK;
}
