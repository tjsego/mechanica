/**
 * @file MxCPotential.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxPotential
 * @date 2022-03-30
 */

#include "MxCPotential.h"

#include "mechanica_c_private.h"
#include "MxCParticle.h"

#include <MxParticle.h>
#include <MxPotential.h>


////////////////////////
// Function factories //
////////////////////////


// MxPotentialEval_ByParticle

static MxPotentialEval_ByParticleHandleFcn _MxPotentialEval_ByParticle_factory_evalFcn;

void MxPotentialEval_ByParticle_eval(MxPotential *p, MxParticle *part_i, float *dx, float r2, float *e, float *f) {
    MxPotentialHandle pHandle {(void*)p};
    MxParticleHandleHandle part_iHandle {(void*)part_i->py_particle()};
    
    (*_MxPotentialEval_ByParticle_factory_evalFcn)(&pHandle, &part_iHandle, dx, r2, e, f);
}

HRESULT MxPotentialEval_ByParticle_factory(struct MxPotentialEval_ByParticleHandle &handle, MxPotentialEval_ByParticleHandleFcn &fcn) {
    _MxPotentialEval_ByParticle_factory_evalFcn = fcn;
    MxPotentialEval_ByParticle *eval_fcn = new MxPotentialEval_ByParticle(MxPotentialEval_ByParticle_eval);
    handle.MxObj = (void*)eval_fcn;
    return S_OK;
}

// MxPotentialEval_ByParticles

static MxPotentialEval_ByParticlesHandleFcn _MxPotentialEval_ByParticles_factory_evalFcn;

void MxPotentialEval_ByParticles_eval(struct MxPotential *p, 
                                      struct MxParticle *part_i, 
                                      struct MxParticle *part_j, 
                                      float *dx, 
                                      float r2, 
                                      float *e, 
                                      float *f) 
{
    MxPotentialHandle pHandle {(void*)p};
    MxParticleHandleHandle part_iHandle {(void*)part_i->py_particle()};
    MxParticleHandleHandle part_jHandle {(void*)part_j->py_particle()};

    (*_MxPotentialEval_ByParticles_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, dx, r2, e, f);
}

HRESULT MxPotentialEval_ByParticles_factory(struct MxPotentialEval_ByParticlesHandle &handle, MxPotentialEval_ByParticlesHandleFcn &fcn) {
    _MxPotentialEval_ByParticles_factory_evalFcn = fcn;
    MxPotentialEval_ByParticles *eval_fcn = new MxPotentialEval_ByParticles(MxPotentialEval_ByParticles_eval);
    handle.MxObj = (void*)eval_fcn;
    return S_OK;
}

// MxPotentialEval_ByParticles3

static MxPotentialEval_ByParticles3HandleFcn _MxPotentialEval_ByParticles3_factory_evalFcn;

void MxPotentialEval_ByParticles3_eval(struct MxPotential *p, 
                                       struct MxParticle *part_i, 
                                       struct MxParticle *part_j, 
                                       struct MxParticle *part_k, 
                                       float ctheta, 
                                       float *e, 
                                       float *fi, 
                                       float *fk) 
{
    MxPotentialHandle pHandle {(void*)p};
    MxParticleHandleHandle part_iHandle {(void*)part_i->py_particle()};
    MxParticleHandleHandle part_jHandle {(void*)part_j->py_particle()};
    MxParticleHandleHandle part_kHandle {(void*)part_k->py_particle()};

    (*_MxPotentialEval_ByParticles3_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, &part_kHandle, ctheta, e, fi, fk);
}

HRESULT MxPotentialEval_ByParticles3_factory(struct MxPotentialEval_ByParticles3Handle &handle, MxPotentialEval_ByParticles3HandleFcn &fcn) {
    _MxPotentialEval_ByParticles3_factory_evalFcn = fcn;
    MxPotentialEval_ByParticles3 *eval_fcn = new MxPotentialEval_ByParticles3(MxPotentialEval_ByParticles3_eval);
    handle.MxObj = (void*)eval_fcn;
    return S_OK;
}

// MxPotentialEval_ByParticles4

static MxPotentialEval_ByParticles4HandleFcn _MxPotentialEval_ByParticles4_factory_evalFcn;

void MxPotentialEval_ByParticles4_eval(struct MxPotential *p, 
                                       struct MxParticle *part_i, 
                                       struct MxParticle *part_j, 
                                       struct MxParticle *part_k, 
                                       struct MxParticle *part_l, 
                                       float cphi, 
                                       float *e, 
                                       float *fi, 
                                       float *fl)
{
    MxPotentialHandle pHandle {(void*)p};
    MxParticleHandleHandle part_iHandle {(void*)part_i->py_particle()};
    MxParticleHandleHandle part_jHandle {(void*)part_j->py_particle()};
    MxParticleHandleHandle part_kHandle {(void*)part_k->py_particle()};
    MxParticleHandleHandle part_lHandle {(void*)part_l->py_particle()};

    (*_MxPotentialEval_ByParticles4_factory_evalFcn)(&pHandle, &part_iHandle, &part_jHandle, &part_kHandle, &part_lHandle, cphi, e, fi, fl);
}

HRESULT MxPotentialEval_ByParticles4_factory(struct MxPotentialEval_ByParticles4Handle &handle, MxPotentialEval_ByParticles4HandleFcn fcn) {
    _MxPotentialEval_ByParticles4_factory_evalFcn = fcn;
    MxPotentialEval_ByParticles4 *eval_fcn = new MxPotentialEval_ByParticles4(MxPotentialEval_ByParticles4_eval);
    handle.MxObj = (void*)eval_fcn;
    return S_OK;
}

// MxPotentialClear

static MxPotentialClearHandleFcn _MxPotentialClear_factory_evalFcn;

void MxPotentialClear_eval(struct MxPotential *p) {
    MxPotentialHandle pHandle {(void*)p};
    (*_MxPotentialClear_factory_evalFcn)(&pHandle);
}

HRESULT MxPotentialClear_factory(struct MxPotentialClearHandle &handle, MxPotentialClearHandleFcn fcn) {
    _MxPotentialClear_factory_evalFcn = fcn;
    MxPotentialClear *eval_fcn = new MxPotentialClear(MxPotentialClear_eval);
    handle.MxObj = (void*)eval_fcn;
    return S_OK;
}

//////////////////
// Module casts //
//////////////////

namespace mx { 

MxPotential *castC(struct MxPotentialHandle *handle) {
    return castC<MxPotential, MxPotentialHandle>(handle);
}

}

#define MXPOTENTIALHANDLE_GET(handle) \
    MxPotential *pot = mx::castC<MxPotential, MxPotentialHandle>(handle); \
    MXCPTRCHECK(pot);


////////////////////
// PotentialFlags //
////////////////////


HRESULT MxCPotentialFlags_init(struct PotentialFlagsHandle *handle) {
    handle->POTENTIAL_NONE = POTENTIAL_NONE;
    handle->POTENTIAL_LJ126 = POTENTIAL_LJ126;
    handle->POTENTIAL_EWALD = POTENTIAL_EWALD;
    handle->POTENTIAL_COULOMB = POTENTIAL_COULOMB;
    handle->POTENTIAL_SINGLE = POTENTIAL_SINGLE;
    handle->POTENTIAL_R2 = POTENTIAL_R2;
    handle->POTENTIAL_R = POTENTIAL_R;
    handle->POTENTIAL_ANGLE = POTENTIAL_ANGLE;
    handle->POTENTIAL_HARMONIC = POTENTIAL_HARMONIC;
    handle->POTENTIAL_DIHEDRAL = POTENTIAL_DIHEDRAL;
    handle->POTENTIAL_SWITCH = POTENTIAL_SWITCH;
    handle->POTENTIAL_REACTIVE = POTENTIAL_REACTIVE;
    handle->POTENTIAL_SCALED = POTENTIAL_SCALED;
    handle->POTENTIAL_SHIFTED = POTENTIAL_SHIFTED;
    handle->POTENTIAL_BOUND = POTENTIAL_BOUND;
    handle->POTENTIAL_SUM = POTENTIAL_SUM;
    handle->POTENTIAL_PERIODIC = POTENTIAL_PERIODIC;
    handle->POTENTIAL_COULOMBR = POTENTIAL_COULOMBR;
    return S_OK;
}


///////////////////
// PotentialKind //
///////////////////


HRESULT MxCPotentialKind_init(struct PotentialKindHandle *handle) {
    handle->POTENTIAL_KIND_POTENTIAL = POTENTIAL_KIND_POTENTIAL;
    handle->POTENTIAL_KIND_DPD = POTENTIAL_KIND_DPD;
    handle->POTENTIAL_KIND_BYPARTICLES = POTENTIAL_KIND_BYPARTICLES;
    handle->POTENTIAL_KIND_COMBINATION = POTENTIAL_KIND_COMBINATION;
    return S_OK;
}


////////////////////////////////
// MxPotentialEval_ByParticle //
////////////////////////////////


HRESULT MxCPotentialEval_ByParticle_init(struct MxPotentialEval_ByParticleHandle *handle, MxPotentialEval_ByParticleHandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxPotentialEval_ByParticle_factory(*handle, *fcn);
}

HRESULT MxCPotentialEval_ByParticle_destroy(struct MxPotentialEval_ByParticleHandle *handle) {
    return mx::capi::destroyHandle<MxPotentialEval_ByParticle, MxPotentialEval_ByParticleHandle>(handle) ? S_OK : E_FAIL;
}

/////////////////////////////////
// MxPotentialEval_ByParticles //
/////////////////////////////////


HRESULT MxCPotentialEval_ByParticles_init(struct MxPotentialEval_ByParticlesHandle *handle, MxPotentialEval_ByParticlesHandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxPotentialEval_ByParticles_factory(*handle, *fcn);
}

HRESULT MxCPotentialEval_ByParticles_destroy(struct MxPotentialEval_ByParticlesHandle *handle) {
    return mx::capi::destroyHandle<MxPotentialEval_ByParticles, MxPotentialEval_ByParticlesHandle>(handle) ? S_OK : E_FAIL;
}


//////////////////////////////////
// MxPotentialEval_ByParticles3 //
//////////////////////////////////


HRESULT MxCPotentialEval_ByParticles3_init(struct MxPotentialEval_ByParticles3Handle *handle, MxPotentialEval_ByParticles3HandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxPotentialEval_ByParticles3_factory(*handle, *fcn);
}

HRESULT MxCPotentialEval_ByParticles3_destroy(struct MxPotentialEval_ByParticles3Handle *handle) {
    return mx::capi::destroyHandle<MxPotentialEval_ByParticles3, MxPotentialEval_ByParticles3Handle>(handle) ? S_OK : E_FAIL;
}


//////////////////////////////////
// MxPotentialEval_ByParticles4 //
//////////////////////////////////


HRESULT MxCPotentialEval_ByParticles4_init(struct MxPotentialEval_ByParticles4Handle *handle, MxPotentialEval_ByParticles4HandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxPotentialEval_ByParticles4_factory(*handle, *fcn);
}

HRESULT MxCPotentialEval_ByParticles4_destroy(struct MxPotentialEval_ByParticles4Handle *handle) {
    return mx::capi::destroyHandle<MxPotentialEval_ByParticles4, MxPotentialEval_ByParticles4Handle>(handle) ? S_OK : E_FAIL;
}


//////////////////////
// MxPotentialClear //
//////////////////////


HRESULT MxCPotentialClear_init(struct MxPotentialClearHandle *handle, MxPotentialClearHandleFcn *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn);
    return MxPotentialClear_factory(*handle, *fcn);
}

HRESULT MxCPotentialClear_destroy(struct MxPotentialClearHandle *handle) {
    return mx::capi::destroyHandle<MxPotentialClear, MxPotentialClearHandle>(handle) ? S_OK : E_FAIL;
}


/////////////////
// MxPotential //
/////////////////


HRESULT MxCPotential_destroy(struct MxPotentialHandle *handle) {
    return mx::capi::destroyHandle<MxPotential, MxPotentialHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCPotential_getName(struct MxPotentialHandle *handle, char **name, unsigned int *numChars) {
    MXPOTENTIALHANDLE_GET(handle);
    return mx::capi::str2Char(std::string(pot->name), name, numChars);;
}

HRESULT MxCPotential_setName(struct MxPotentialHandle *handle, const char *name) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(name);
    std::string sname(name);
    char *cname = new char[sname.size() + 1];
    std::strcpy(cname, sname.c_str());
    pot->name = cname;
    return S_OK;
}

HRESULT MxCPotential_getFlags(struct MxPotentialHandle *handle, unsigned int *flags) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(flags);
    *flags = pot->flags;
    return S_OK;
}

HRESULT MxCPotential_setFlags(struct MxPotentialHandle *handle, unsigned int flags) {
    MXPOTENTIALHANDLE_GET(handle);
    pot->flags = flags;
    return S_OK;
}

HRESULT MxCPotential_getKind(struct MxPotentialHandle *handle, unsigned int *kind) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(kind);
    *kind = pot->kind;
    return S_OK;
}

HRESULT MxCPotential_evalR(struct MxPotentialHandle *handle, float r, float *potE) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(potE);
    *potE = (*pot)(r);
    return S_OK;
}

HRESULT MxCPotential_evalR0(struct MxPotentialHandle *handle, float r, float r0, float *potE) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(potE);
    *potE = (*pot)(r, r0);
    return S_OK;
}

HRESULT MxCPotential_evalPos(struct MxPotentialHandle *handle, float *pos, float *potE) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(potE);
    *potE = (*pot)(std::vector<float>{pos[0], pos[1], pos[2]});
    return S_OK;
}

HRESULT MxCPotential_evalPart(struct MxPotentialHandle *handle, struct MxParticleHandleHandle *partHandle, float *pos, float *potE) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(partHandle); MXCPTRCHECK(partHandle->MxObj);
    MXCPTRCHECK(potE);
    MxParticleHandle *ph = (MxParticleHandle*)partHandle->MxObj;
    *potE = (*pot)(ph, MxVector3f::from(pos));
    return S_OK;
}

HRESULT MxCPotential_evalParts2(struct MxPotentialHandle *handle, 
                                struct MxParticleHandleHandle *phi, 
                                struct MxParticleHandleHandle *phj, 
                                float *potE) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(potE);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    *potE = (*pot)(pi, pj);
    return S_OK;
}

HRESULT MxCPotential_evalParts3(struct MxPotentialHandle *handle, 
                                struct MxParticleHandleHandle *phi, 
                                struct MxParticleHandleHandle *phj, 
                                struct MxParticleHandleHandle *phk, 
                                float *potE) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(phk); MXCPTRCHECK(phk->MxObj);
    MXCPTRCHECK(potE);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    MxParticleHandle *pk = (MxParticleHandle*)phk->MxObj;
    *potE = (*pot)(pi, pj, pk);
    return S_OK;
}

HRESULT MxCPotential_evalParts4(struct MxPotentialHandle *handle, 
                                struct MxParticleHandleHandle *phi, 
                                struct MxParticleHandleHandle *phj, 
                                struct MxParticleHandleHandle *phk, 
                                struct MxParticleHandleHandle *phl, 
                                float *potE) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(phk); MXCPTRCHECK(phk->MxObj);
    MXCPTRCHECK(phl); MXCPTRCHECK(phl->MxObj);
    MXCPTRCHECK(potE);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    MxParticleHandle *pk = (MxParticleHandle*)phk->MxObj;
    MxParticleHandle *pl = (MxParticleHandle*)phl->MxObj;
    *potE = (*pot)(pi, pj, pk, pl);
    return S_OK;
}

HRESULT MxCPotential_fevalR(struct MxPotentialHandle *handle, float r, float *force) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(force);
    *force = pot->force(r);
    return S_OK;
}

HRESULT MxCPotential_fevalR0(struct MxPotentialHandle *handle, float r, float r0, float *force) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(force);
    *force = pot->force(r, r0);
    return S_OK;
}

HRESULT MxCPotential_fevalPos(struct MxPotentialHandle *handle, float *pos, float **force) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(pos);
    MXCPTRCHECK(force);
    std::vector<float> _pos {pos[0], pos[1], pos[2]};
    MxVector3f f = MxVector3f::from(pot->force(_pos).data());
    MXVECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT MxCPotential_fevalPart(struct MxPotentialHandle *handle, struct MxParticleHandleHandle *partHandle, float *pos, float **force) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(partHandle); MXCPTRCHECK(partHandle->MxObj);
    MXCPTRCHECK(pos);
    MXCPTRCHECK(force);
    MxParticleHandle *ph = (MxParticleHandle*)partHandle->MxObj;
    MxVector3f f = MxVector3f::from(pot->force(ph, MxVector3f::from(pos)).data());
    MXVECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT MxCPotential_fevalParts2(struct MxPotentialHandle *handle, 
                                 struct MxParticleHandleHandle *phi, 
                                 struct MxParticleHandleHandle *phj, 
                                 float **force) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(force);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    MxVector3f f = MxVector3f::from(pot->force(pi, pj).data());
    MXVECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT MxCPotential_fevalParts3(struct MxPotentialHandle *handle, 
                                 struct MxParticleHandleHandle *phi, 
                                 struct MxParticleHandleHandle *phj, 
                                 struct MxParticleHandleHandle *phk, 
                                 float **forcei, 
                                 float **forcek) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(phk); MXCPTRCHECK(phk->MxObj);
    MXCPTRCHECK(forcei);
    MXCPTRCHECK(forcek);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    MxParticleHandle *pk = (MxParticleHandle*)phk->MxObj;
    std::vector<float> fi, fk;
    std::tie(fi, fk) = pot->force(pi, pj, pk);
    MxVector3f _fi = MxVector3f::from(fi.data());
    MxVector3f _fk = MxVector3f::from(fk.data());
    MXVECTOR3_COPYFROM(_fi, (*forcei));
    MXVECTOR3_COPYFROM(_fk, (*forcek));
    return S_OK;
}

HRESULT MxCPotential_fevalParts4(struct MxPotentialHandle *handle, 
                                 struct MxParticleHandleHandle *phi, 
                                 struct MxParticleHandleHandle *phj, 
                                 struct MxParticleHandleHandle *phk, 
                                 struct MxParticleHandleHandle *phl, 
                                 float **forcei, 
                                 float **forcel) 
{
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(phi); MXCPTRCHECK(phi->MxObj);
    MXCPTRCHECK(phj); MXCPTRCHECK(phj->MxObj);
    MXCPTRCHECK(phk); MXCPTRCHECK(phk->MxObj);
    MXCPTRCHECK(phl); MXCPTRCHECK(phl->MxObj);
    MXCPTRCHECK(forcei);
    MXCPTRCHECK(forcel);
    MxParticleHandle *pi = (MxParticleHandle*)phi->MxObj;
    MxParticleHandle *pj = (MxParticleHandle*)phj->MxObj;
    MxParticleHandle *pk = (MxParticleHandle*)phk->MxObj;
    MxParticleHandle *pl = (MxParticleHandle*)phl->MxObj;
    std::vector<float> fi, fl;
    std::tie(fi, fl) = pot->force(pi, pj, pk, pl);
    MxVector3f _fi = MxVector3f::from(fi.data());
    MxVector3f _fl = MxVector3f::from(fl.data());
    MXVECTOR3_COPYFROM(_fi, (*forcei));
    MXVECTOR3_COPYFROM(_fl, (*forcel));
    return S_OK;
}

HRESULT MxCPotential_getConstituents(struct MxPotentialHandle *handle, struct MxPotentialHandle ***chandles, unsigned int *numPots) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(chandles);
    MXCPTRCHECK(numPots);
    auto constituents = pot->constituents();
    *numPots = constituents.size();
    if(*numPots > 0) {
        MxPotentialHandle **_chandles = (MxPotentialHandle**)malloc(sizeof(MxPotentialHandle*) * *numPots);
        if(!_chandles) 
            return E_OUTOFMEMORY;
        MxPotentialHandle *ph;
        for(unsigned int i = 0; i < *numPots; i++) {
            ph = new MxPotentialHandle();
            ph->MxObj = (void*)constituents[i];
            _chandles[i] = ph;
        }
        *chandles = _chandles;
    }
    return S_OK;
}

HRESULT MxCPotential_toString(struct MxPotentialHandle *handle, char **str, unsigned int *numChars) {
    MXPOTENTIALHANDLE_GET(handle);
    return mx::capi::str2Char(pot->toString(), str, numChars);;
}

HRESULT MxCPotential_fromString(struct MxPotentialHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxPotential *pot = MxPotential::fromString(str);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_setClearFcn(struct MxPotentialHandle *handle, struct MxPotentialClearHandle *fcn) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    pot->clear_func = *(MxPotentialClear*)fcn->MxObj;
    return S_OK;
}

HRESULT MxCPotential_removeClearFcn(struct MxPotentialHandle *handle) {
    MXPOTENTIALHANDLE_GET(handle);
    if(pot->clear_func) 
        pot->clear_func = NULL;
    return S_OK;
}

HRESULT MxCPotential_hasClearFcn(struct MxPotentialHandle *handle, bool *hasClear) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(hasClear);
    *hasClear = pot->clear_func != NULL;
    return S_OK;
}

HRESULT MxCPotential_getMin(struct MxPotentialHandle *handle, float *minR) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(minR);
    *minR = pot->getMin();
    return S_OK;
}

HRESULT MxCPotential_getMax(struct MxPotentialHandle *handle, float *maxR) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(maxR);
    *maxR = pot->getMax();
    return S_OK;
}

HRESULT MxCPotential_getBound(struct MxPotentialHandle *handle, bool *bound) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(bound);
    *bound = pot->getBound();
    return S_OK;
}

HRESULT MxCPotential_setBound(struct MxPotentialHandle *handle, bool bound) {
    MXPOTENTIALHANDLE_GET(handle);
    pot->setBound(bound);
    return S_OK;
}

HRESULT MxCPotential_getR0(struct MxPotentialHandle *handle, float *r0) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(r0);
    *r0 = pot->getR0();
    return S_OK;
}

HRESULT MxCPotential_setR0(struct MxPotentialHandle *handle, float r0) {
    MXPOTENTIALHANDLE_GET(handle);
    pot->setR0(r0);
    return S_OK;
}

HRESULT MxCPotential_getRSquare(struct MxPotentialHandle *handle, float *r2) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(r2);
    *r2 = pot->getRSquare();
    return S_OK;
}

HRESULT MxCPotential_getShifted(struct MxPotentialHandle *handle, bool *shifted) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(shifted);
    *shifted = pot->getShifted();
    return S_OK;
}

HRESULT MxCPotential_getPeriodic(struct MxPotentialHandle *handle, bool *periodic) {
    MXPOTENTIALHANDLE_GET(handle);
    MXCPTRCHECK(periodic);
    *periodic = pot->getPeriodic();
    return S_OK;
}

HRESULT MxCPotential_create_lennard_jones_12_6(struct MxPotentialHandle *handle, double min, double max, double A, double B, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::lennard_jones_12_6(min, max, A, B, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_lennard_jones_12_6_coulomb(struct MxPotentialHandle *handle, double min, double max, double A, double B, double q, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::lennard_jones_12_6_coulomb(min, max, A, B, q, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_ewald(struct MxPotentialHandle *handle, double min, double max, double q, double kappa, double *tol, unsigned int *periodicOrder) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::ewald(min, max, q, kappa, tol, periodicOrder);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_coulomb(struct MxPotentialHandle *handle, double q, double *min, double *max, double *tol, unsigned int *periodicOrder) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::coulomb(q, min, max, tol, periodicOrder);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_coulombR(struct MxPotentialHandle *handle, double q, double kappa, double min, double max, unsigned int* modes) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::coulombR(q, kappa, min, max, modes);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_harmonic(struct MxPotentialHandle *handle, double k, double r0, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::harmonic(k, r0, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_linear(struct MxPotentialHandle *handle, double k, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::linear(k, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_harmonic_angle(struct MxPotentialHandle *handle, double k, double theta0, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::harmonic_angle(k, theta0, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_harmonic_dihedral(struct MxPotentialHandle *handle, double k, double delta, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::harmonic_dihedral(k, delta, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_cosine_dihedral(struct MxPotentialHandle *handle, double k, int n, double delta, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::cosine_dihedral(k, n, delta, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_well(struct MxPotentialHandle *handle, double k, double n, double r0, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::well(k, n, r0, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_glj(struct MxPotentialHandle *handle, double e, double *m, double *n, double *k, double *r0, double *min, double *max, double *tol, bool *shifted) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::glj(e, m, n, k, r0, min, max, tol, shifted);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_morse(struct MxPotentialHandle *handle, double *d, double *a, double *r0, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::morse(d, a, r0, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_overlapping_sphere(struct MxPotentialHandle *handle, double *mu, double *kc, double *kh, double *r0, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::overlapping_sphere(mu, kc, kh, r0, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_power(struct MxPotentialHandle *handle, double *k, double *r0, double *alpha, double *min, double *max, double *tol) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::power(k, r0, alpha, min, max, tol);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_dpd(struct MxPotentialHandle *handle, double *alpha, double *gamma, double *sigma, double *cutoff, bool *shifted) {
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::dpd(alpha, gamma, sigma, cutoff, shifted);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_custom(struct MxPotentialHandle *handle, double min, double max, double (*f)(double), double (*fp)(double), double (*f6p)(double),
                            double *tol, unsigned int *flags) 
{
    MXCPTRCHECK(handle);
    MxPotential *pot = MxPotential::custom(min, max, f, fp, f6p, tol, flags);
    MXCPTRCHECK(pot);
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_eval_ByParticle(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticleHandle *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    MxPotential *pot = new MxPotential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_bypart = (MxPotentialEval_ByParticle)fcn->MxObj;
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_eval_ByParticles(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticlesHandle *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    MxPotential *pot = new MxPotential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts = (MxPotentialEval_ByParticles)fcn->MxObj;
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_eval_ByParticles3(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticles3Handle *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    MxPotential *pot = new MxPotential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts3 = (MxPotentialEval_ByParticles3)fcn->MxObj;
    handle->MxObj = (void*)pot;
    return S_OK;
}

HRESULT MxCPotential_create_eval_ByParticles4(struct MxPotentialHandle *handle, struct MxPotentialEval_ByParticles4Handle *fcn) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(fcn); MXCPTRCHECK(fcn->MxObj);
    MxPotential *pot = new MxPotential();
    pot->kind = POTENTIAL_KIND_BYPARTICLES;
    pot->eval_byparts4 = (MxPotentialEval_ByParticles4)fcn->MxObj;
    handle->MxObj = (void*)pot;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Add two potentials
 * 
 * @param handlei first potential
 * @param handlej second potential
 * @param handleSum resulting potential
 * @return HRESULT 
 */
HRESULT MxCPotential_add(struct MxPotentialHandle *handlei, struct MxPotentialHandle *handlej, struct MxPotentialHandle *handleSum) {
    MXCPTRCHECK(handlei); MXCPTRCHECK(handlei->MxObj);
    MXCPTRCHECK(handlej); MXCPTRCHECK(handlej->MxObj);
    MXCPTRCHECK(handleSum); MXCPTRCHECK(handleSum->MxObj);
    MxPotential *poti = (MxPotential*)handlei->MxObj;
    MxPotential *potj = (MxPotential*)handlej->MxObj;
    MxPotential *potk = &(*poti + *potj);
    handleSum->MxObj = (void*)(potk);
    return S_OK;
}
