/**
 * @file MxCFlux.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxFluxes
 * @date 2022-04-03
 */

#include "MxCFlux.h"

#include "mechanica_c_private.h"

#include <Flux.hpp>

namespace mx { 

MxFlux *castC(struct MxFluxHandle *handle) {
    return castC<MxFlux, MxFluxHandle>(handle);
}

MxFluxes *castC(struct MxFluxesHandle *handle) {
    return castC<MxFluxes, MxFluxesHandle>(handle);
}

}

#define MXFLUX_GET(handle, varname) \
    MxFlux *varname = mx::castC<MxFlux, MxFluxHandle>(handle); \
    MXCPTRCHECK(varname);

#define MXFLUXES_GET(handle, varname) \
    MxFluxes *varname = mx::castC<MxFluxes, MxFluxesHandle>(handle); \
    MXCPTRCHECK(varname);


//////////////
// FluxKind //
//////////////


HRESULT MxCFluxKindHandle_init(struct MxFluxKindHandle *handle) {
    MXCPTRCHECK(handle);
    handle->FLUX_FICK = FLUX_FICK;
    handle->FLUX_SECRETE = FLUX_SECRETE;
    handle->FLUX_UPTAKE = FLUX_UPTAKE;
    return S_OK;
}


////////////
// MxFlux //
////////////


HRESULT MxCFlux_getSize(struct MxFluxHandle *handle, unsigned int *size) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(size);
    *size = flx->size;
    return S_OK;
}

HRESULT MxCFlux_getKind(struct MxFluxHandle *handle, unsigned int index, unsigned int *kind) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(kind);
    if(index >= flx->size) 
        return E_FAIL;
    *kind = flx->kinds[index];
    return S_OK;
}

HRESULT MxCFlux_getTypeIds(struct MxFluxHandle *handle, unsigned int index, unsigned int *typeid_a, unsigned int *typeid_b) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(typeid_a);
    MXCPTRCHECK(typeid_b);
    if(index >= flx->size) 
        return E_FAIL;
    auto typeids = flx->type_ids[index];
    *typeid_a = typeids.a;
    *typeid_b = typeids.b;
    return S_OK;
}

HRESULT MxCFlux_getCoef(struct MxFluxHandle *handle, unsigned int index, float *coef) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(coef);
    if(index >= flx->size) 
        return E_FAIL;
    *coef = flx->coef[index];
    return S_OK;
}

HRESULT MxCFlux_setCoef(struct MxFluxHandle *handle, unsigned int index, float coef) {
    MXFLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->coef[index] = coef;
    return S_OK;
}

HRESULT MxCFlux_getDecayCoef(struct MxFluxHandle *handle, unsigned int index, float *decay_coef) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(decay_coef);
    if(index >= flx->size) 
        return E_FAIL;
    *decay_coef = flx->decay_coef[index];
    return S_OK;
}

HRESULT MxCFlux_setDecayCoef(struct MxFluxHandle *handle, unsigned int index, float decay_coef) {
    MXFLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->decay_coef[index] = decay_coef;
    return S_OK;
}

HRESULT MxCFlux_getTarget(struct MxFluxHandle *handle, unsigned int index, float *target) {
    MXFLUX_GET(handle, flx);
    MXCPTRCHECK(target);
    if(index >= flx->size) 
        return E_FAIL;
    *target = flx->target[index];
    return S_OK;
}

HRESULT MxCFlux_setTarget(struct MxFluxHandle *handle, unsigned int index, float target) {
    MXFLUX_GET(handle, flx);
    if(index >= flx->size) 
        return E_FAIL;
    flx->target[index] = target;
    return S_OK;
}


//////////////
// MxFluxes //
//////////////


HRESULT MxCFluxes_getSize(struct MxFluxesHandle *handle, int *size) {
    MXFLUXES_GET(handle, flxs);
    MXCPTRCHECK(size);
    *size = flxs->size;
    return S_OK;
}

HRESULT MxCFluxes_getFlux(struct MxFluxesHandle *handle, unsigned int index, struct MxFluxHandle *flux) {
    MXFLUXES_GET(handle, flxs);
    MXCPTRCHECK(flux);
    if(index >= flxs->size) 
        return E_FAIL;
    flux->MxObj = (void*)&flxs->fluxes[index];
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////

HRESULT MxCFluxes_fluxFick(struct MxFluxesHandle *handle, 
                           struct MxParticleTypeHandle *A, 
                           struct MxParticleTypeHandle *B, 
                           const char *name, 
                           float k, 
                           float decay) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(A); MXCPTRCHECK(A->MxObj);
    MXCPTRCHECK(B); MXCPTRCHECK(B->MxObj);
    MXCPTRCHECK(name);
    MxFluxes *flxs = MxFluxes::fluxFick((MxParticleType*)A->MxObj, (MxParticleType*)B->MxObj, name, k, decay);
    MXCPTRCHECK(flxs);
    handle->MxObj = (void*)flxs;
    return S_OK;
}

HRESULT MxCFluxes_secrete(struct MxFluxesHandle *handle, 
                          struct MxParticleTypeHandle *A, 
                          struct MxParticleTypeHandle *B, 
                          const char *name, 
                          float k, 
                          float target, 
                          float decay) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(A); MXCPTRCHECK(A->MxObj);
    MXCPTRCHECK(B); MXCPTRCHECK(B->MxObj);
    MXCPTRCHECK(name);
    MxFluxes *flxs = MxFluxes::secrete((MxParticleType*)A->MxObj, (MxParticleType*)B->MxObj, name, k, target, decay);
    MXCPTRCHECK(flxs);
    handle->MxObj = (void*)flxs;
    return S_OK;
}

HRESULT MxCFluxes_uptake(struct MxFluxesHandle *handle, 
                         struct MxParticleTypeHandle *A, 
                         struct MxParticleTypeHandle *B, 
                         const char *name, 
                         float k, 
                         float target, 
                         float decay) 
{
    MXCPTRCHECK(handle);
    MXCPTRCHECK(A); MXCPTRCHECK(A->MxObj);
    MXCPTRCHECK(B); MXCPTRCHECK(B->MxObj);
    MXCPTRCHECK(name);
    MxFluxes *flxs = MxFluxes::uptake((MxParticleType*)A->MxObj, (MxParticleType*)B->MxObj, name, k, target, decay);
    MXCPTRCHECK(flxs);
    handle->MxObj = (void*)flxs;
    return S_OK;
}
