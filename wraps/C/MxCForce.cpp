/**
 * @file MxCForce.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxForce
 * @date 2022-03-30
 */

#include "MxCForce.h"

#include "mechanica_c_private.h"
#include "MxCParticle.h"

#include <MxForce.h>
#include <MxParticle.h>


////////////////////////
// Function factories //
////////////////////////

// MxUserForceFuncType

static MxUserForceFuncTypeHandleFcn _MxUserForceFuncType_factory_evalFcn;

MxVector3f MxUserForceFuncType_eval(MxConstantForce *f) {
    MxConstantForceHandle fhandle {(void*)f};
    MxVector3f res;
    _MxUserForceFuncType_factory_evalFcn(&fhandle, res.data());
    return res;
}

HRESULT MxUserForceFuncType_factory(struct MxUserForceFuncTypeHandle *handle, MxUserForceFuncTypeHandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    _MxUserForceFuncType_factory_evalFcn = *fcn;
    MxUserForceFuncType *eval_fcn = new MxUserForceFuncType(MxUserForceFuncType_eval);
    handle->MxObj = (void*)eval_fcn;
    return S_OK;
}


//////////////////
// Module casts //
//////////////////

namespace mx { 

MxForce *castC(struct MxForceHandle *handle) {
    return castC<MxForce, MxForceHandle>(handle);
}

MxForceSum *castC(struct MxForceSumHandle *handle) {
    return castC<MxForceSum, MxForceSumHandle>(handle);
}

MxConstantForce *castC(struct MxConstantForceHandle *handle) {
    return castC<MxConstantForce, MxConstantForceHandle>(handle);
}

Berendsen *castC(struct BerendsenHandle *handle) {
    return castC<Berendsen, BerendsenHandle>(handle);
}

Gaussian *castC(struct GaussianHandle *handle) {
    return castC<Gaussian, GaussianHandle>(handle);
}

Friction *castC(struct FrictionHandle *handle) {
    return castC<Friction, FrictionHandle>(handle);
}

}

#define MXFORCEHANDLE_GET(handle) \
    MxForce *frc = mx::castC<MxForce, MxForceHandle>(handle); \
    MXCPTRCHECK(frc);

#define MXFORCESUMHANDLE_GET(handle) \
    MxForceSum *frc = mx::castC<MxForceSum, MxForceSumHandle>(handle); \
    MXCPTRCHECK(frc);

#define MXCONSTANTFORCEHANDLE_GET(handle) \
    MxConstantForce *frc = mx::castC<MxConstantForce, MxConstantForceHandle>(handle); \
    MXCPTRCHECK(frc);

#define BERENDSENHANDLE_GET(handle) \
    Berendsen *frc = mx::castC<Berendsen, BerendsenHandle>(handle); \
    MXCPTRCHECK(frc);

#define GAUSSIANHANDLE_GET(handle) \
    Gaussian *frc = mx::castC<Gaussian, GaussianHandle>(handle); \
    MXCPTRCHECK(frc);

#define FRICTIONHANDLE_GET(handle) \
    Friction *frc = mx::castC<Friction, FrictionHandle>(handle); \
    MXCPTRCHECK(frc);


//////////////////
// MXFORCE_TYPE //
//////////////////


HRESULT MxCFORCE_TYPE_init(struct MXFORCE_TYPEHandle *handle) {
    handle->FORCE_FORCE = FORCE_FORCE;
    handle->FORCE_BERENDSEN = FORCE_BERENDSEN;
    handle->FORCE_GAUSSIAN = FORCE_GAUSSIAN;
    handle->FORCE_FRICTION = FORCE_FRICTION;
    handle->FORCE_SUM = FORCE_SUM;
    handle->FORCE_CONSTANT = FORCE_CONSTANT;
    return S_OK;
}


/////////////////////////
// MxUserForceFuncType //
/////////////////////////


HRESULT MxCForce_EvalFcn_init(struct MxUserForceFuncTypeHandle *handle, MxUserForceFuncTypeHandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxUserForceFuncType_factory(handle, fcn);
}

HRESULT MxCForce_EvalFcn_destroy(struct MxUserForceFuncTypeHandle *handle) {
    return mx::capi::destroyHandle<MxUserForceFuncType, MxUserForceFuncTypeHandle>(handle) ? S_OK : E_FAIL;
}


/////////////
// MxForce //
/////////////

HRESULT MxCForce_getType(struct MxForceHandle *handle, unsigned int *te) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(te);
    *te = (unsigned int)frc->type;
    return S_OK;
}

HRESULT MxCForce_bind_species(struct MxForceHandle *handle, struct MxParticleTypeHandle *a_type, const char *coupling_symbol) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(a_type); MXCPTRCHECK(a_type->MxObj);
    MXCPTRCHECK(coupling_symbol);
    MxParticleType *_a_type = (MxParticleType*)a_type->MxObj;
    return frc->bind_species(_a_type, coupling_symbol);
}

HRESULT MxCForce_toString(struct MxForceHandle *handle, char **str, unsigned int *numChars) {
    MXFORCEHANDLE_GET(handle);
    return mx::capi::str2Char(frc->toString(), str, numChars);
}

HRESULT MxCForce_fromString(struct MxForceHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxForce *frc = MxForce::fromString(str);
    MXCPTRCHECK(frc);
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCForce_destroy(struct MxForceHandle *handle) {
    return mx::capi::destroyHandle<MxForce, MxForceHandle>(handle) ? S_OK : E_FAIL;
}


////////////////
// MxForceSum //
////////////////


HRESULT MxCForceSum_checkType(struct MxForceHandle *handle, bool *isType) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(isType);
    *isType = frc->type == FORCE_SUM;
    return S_OK;
}

HRESULT MxCForceSum_toBase(struct MxForceSumHandle *handle, struct MxForceHandle *baseHandle) {
    MXFORCESUMHANDLE_GET(handle);
    MXCPTRCHECK(baseHandle);
    baseHandle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCForceSum_fromBase(struct MxForceHandle *baseHandle, struct MxForceSumHandle *handle) {
    MXFORCEHANDLE_GET(baseHandle);
    MXCPTRCHECK(handle);
    bool checkType;
    if((MxCForceSum_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCForceSum_getConstituents(struct MxForceSumHandle *handle, struct MxForceHandle *f1, struct MxForceHandle *f2) {
    MXFORCESUMHANDLE_GET(handle);
    MXCPTRCHECK(f1);
    MXCPTRCHECK(f2);
    if(frc->f1) 
        f1->MxObj = (void*)frc->f1;
    if(frc->f2) 
        f2->MxObj = (void*)frc->f2;
    return S_OK;
}


/////////////////////
// MxConstantForce //
/////////////////////


HRESULT MxCConstantForce_init(struct MxConstantForceHandle *handle, struct MxUserForceFuncTypeHandle *func, float period) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(func); MXCPTRCHECK(func->MxObj);
    MxUserForceFuncType *_func = (MxUserForceFuncType*)func->MxObj;
    MxConstantForce *frc = new MxConstantForce(_func, period);
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCConstantForce_checkType(struct MxForceHandle *handle, bool *isType) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(isType);
    *isType = frc->type == FORCE_CONSTANT;
    return S_OK;
}

HRESULT MxCConstantForce_toBase(struct MxConstantForceHandle *handle, struct MxForceHandle *baseHandle) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(baseHandle);
    baseHandle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCConstantForce_fromBase(struct MxForceHandle *baseHandle, struct MxConstantForceHandle *handle) {
    MXFORCEHANDLE_GET(baseHandle);
    MXCPTRCHECK(handle);
    bool checkType;
    if((MxCConstantForce_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCConstantForce_getPeriod(struct MxConstantForceHandle *handle, float *period) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(period);
    *period = frc->getPeriod();
    return S_OK;
}

HRESULT MxCConstantForce_setPeriod(struct MxConstantForceHandle *handle, float period) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    frc->setPeriod(period);
    return S_OK;
}

HRESULT MxCConstantForce_setFunction(struct MxConstantForceHandle *handle, struct MxUserForceFuncTypeHandle *fcn) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    MxUserForceFuncType *_fcn = (MxUserForceFuncType*)fcn->MxObj;
    frc->setValue(_fcn);
    return S_OK;
}

HRESULT MxCConstantForce_getValue(struct MxConstantForceHandle *handle, float **force) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(force);
    MxVector3f _force = frc->getValue();
    MXVECTOR3_COPYFROM(_force, (*force));
    return S_OK;
}

HRESULT MxCConstantForce_setValue(struct MxConstantForceHandle *handle, float *force) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(force);
    frc->setValue(MxVector3f::from(force));
    return S_OK;
}

HRESULT MxCConstantForce_getLastUpdate(struct MxConstantForceHandle *handle, float *lastUpdate) {
    MXCONSTANTFORCEHANDLE_GET(handle);
    MXCPTRCHECK(lastUpdate);
    *lastUpdate = frc->lastUpdate;
    return S_OK;
}


///////////////
// Berendsen //
///////////////


HRESULT MxCBerendsen_init(struct BerendsenHandle *handle, float tau) {
    MXCPTRCHECK(handle);
    MxForce *frc = MxForce::berendsen_tstat(tau);
    MXCPTRCHECK(frc);
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCBerendsen_checkType(struct MxForceHandle *handle, bool *isType) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(isType);
    *isType = frc->type == FORCE_BERENDSEN;
    return S_OK;
}

HRESULT MxCBerendsen_toBase(struct BerendsenHandle *handle, struct MxForceHandle *baseHandle) {
    BERENDSENHANDLE_GET(handle);
    MXCPTRCHECK(baseHandle);
    baseHandle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCBerendsen_fromBase(struct MxForceHandle *baseHandle, struct BerendsenHandle *handle) {
    MXFORCEHANDLE_GET(baseHandle);
    MXCPTRCHECK(handle);
    bool checkType;
    if((MxCBerendsen_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCBerendsen_getTimeConstant(struct BerendsenHandle *handle, float *tau) {
    BERENDSENHANDLE_GET(handle);
    MXCPTRCHECK(tau);
    if(frc->itau == 0.f) 
        return E_FAIL;
    *tau = 1.f / frc->itau;
    return S_OK;
}

HRESULT MxCBerendsen_setTimeConstant(struct BerendsenHandle *handle, float tau) {
    BERENDSENHANDLE_GET(handle);
    if(tau == 0.f) 
        return E_FAIL;
    frc->itau = 1.f / tau;
    return S_OK;
}


//////////////
// Gaussian //
//////////////

HRESULT MxCGaussian_init(struct GaussianHandle *handle, float std, float mean, float duration) {
    MXCPTRCHECK(handle);
    MxForce *frc = MxForce::random(std, mean, duration);
    MXCPTRCHECK(frc);
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCGaussian_checkType(struct MxForceHandle *handle, bool *isType) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(isType);
    *isType = frc->type == FORCE_GAUSSIAN;
    return S_OK;
}

HRESULT MxCGaussian_toBase(struct GaussianHandle *handle, struct MxForceHandle *baseHandle) {
    GAUSSIANHANDLE_GET(handle);
    MXCPTRCHECK(baseHandle);
    baseHandle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCGaussian_fromBase(struct MxForceHandle *baseHandle, struct GaussianHandle *handle) {
    MXFORCEHANDLE_GET(baseHandle);
    MXCPTRCHECK(handle);
    bool checkType;
    if((MxCGaussian_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCGaussian_getStd(struct GaussianHandle *handle, float *std) {
    GAUSSIANHANDLE_GET(handle);
    MXCPTRCHECK(std);
    *std = frc->std;
    return S_OK;
}

HRESULT MxCGaussian_setStd(struct GaussianHandle *handle, float std) {
    GAUSSIANHANDLE_GET(handle);
    frc->std = std;
    return S_OK;
}

HRESULT MxCGaussian_getMean(struct GaussianHandle *handle, float *mean) {
    GAUSSIANHANDLE_GET(handle);
    MXCPTRCHECK(mean);
    *mean = frc->mean;
    return S_OK;
}

HRESULT MxCGaussian_setMean(struct GaussianHandle *handle, float mean) {
    GAUSSIANHANDLE_GET(handle);
    frc->mean = mean;
    return S_OK;
}

HRESULT MxCGaussian_getDuration(struct GaussianHandle *handle, float *duration) {
    GAUSSIANHANDLE_GET(handle);
    MXCPTRCHECK(duration);
    *duration = frc->durration_steps;
    return S_OK;
}

HRESULT MxCGaussian_setDuration(struct GaussianHandle *handle, float duration) {
    GAUSSIANHANDLE_GET(handle);
    frc->durration_steps = duration;
    return S_OK;
}


//////////////
// Friction //
//////////////


HRESULT MxCFriction_init(struct FrictionHandle *handle, float coeff) {
    MXCPTRCHECK(handle);
    MxForce *frc = MxForce::friction(coeff);
    MXCPTRCHECK(frc);
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCFriction_checkType(struct MxForceHandle *handle, bool *isType) {
    MXFORCEHANDLE_GET(handle);
    MXCPTRCHECK(isType);
    *isType = frc->type == FORCE_FRICTION;
    return S_OK;
}

HRESULT MxCFriction_toBase(struct FrictionHandle *handle, struct MxForceHandle *baseHandle) {
    FRICTIONHANDLE_GET(handle);
    MXCPTRCHECK(baseHandle);
    baseHandle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCFriction_fromBase(struct MxForceHandle *baseHandle, struct FrictionHandle *handle) {
    MXFORCEHANDLE_GET(baseHandle);
    MXCPTRCHECK(handle);
    bool checkType;
    if((MxCFriction_checkType(baseHandle, &checkType)) != S_OK) 
        return E_FAIL;
    if(!checkType) 
        return E_FAIL;
    handle->MxObj = (void*)frc;
    return S_OK;
}

HRESULT MxCFriction_getCoef(struct FrictionHandle *handle, float *coef) {
    FRICTIONHANDLE_GET(handle);
    MXCPTRCHECK(coef);
    *coef = frc->coef;
    return S_OK;
}

HRESULT MxCFriction_setCoef(struct FrictionHandle *handle, float coef) {
    FRICTIONHANDLE_GET(handle);
    frc->coef = coef;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCForce_add(struct MxForceHandle *f1, struct MxForceHandle *f2, struct MxForceSumHandle *fSum) {
    MXCPTRCHECK(f1); MXCPTRCHECK(f1->MxObj);
    MXCPTRCHECK(f2); MXCPTRCHECK(f2->MxObj);
    MXCPTRCHECK(fSum);
    MxForce &_f1 = *(MxForce*)f1->MxObj;
    MxForce &_f2 = *(MxForce*)f2->MxObj;
    MxForce &_f3 = _f1 + _f2;
    MxForceSum *_fSum = (MxForceSum*)&_f3;
    fSum->MxObj = (void*)_fSum;
    return S_OK;
}
