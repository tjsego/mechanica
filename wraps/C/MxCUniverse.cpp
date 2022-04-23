/**
 * @file MxCUniverse.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxUniverse
 * @date 2022-03-28
 */

#include "MxCUniverse.h"

#include "mechanica_c_private.h"

#include "MxUniverse.h"

namespace mx {

MxUniverseConfig *castC(struct MxUniverseConfigHandle *handle) {
    return castC<MxUniverseConfig, MxUniverseConfigHandle>(handle);
}

}

#define MXUNIVERSECONFIG_GET(handle) \
    MxUniverseConfig *conf = mx::castC<MxUniverseConfig, MxUniverseConfigHandle>(handle); \
    MXCPTRCHECK(conf);

#define MXUNIVERSE_STATIC_GET() \
    MxUniverse *univ = MxUniverse::get(); \
    MXCPTRCHECK(univ);


////////////////
// MxUniverse //
////////////////


HRESULT MxCUniverse_getOrigin(float **origin) {
    MXUNIVERSE_STATIC_GET()
    auto o = univ->origin();
    MXVECTOR3_COPYFROM(o, (*origin));
    return S_OK;
}

HRESULT MxCUniverse_getDim(float **dim) {
    MXUNIVERSE_STATIC_GET()
    auto d = univ->dim();
    MXVECTOR3_COPYFROM(d, (*dim));
    return S_OK;
}

HRESULT MxCUniverse_getIsRunning(bool *isRunning) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(isRunning);
    *isRunning = univ->isRunning;
    return S_OK;
}

HRESULT MxCUniverse_getName(char **name, unsigned int *numChars) {
    MXUNIVERSE_STATIC_GET()
    return mx::capi::str2Char(univ->name, name, numChars);
}

HRESULT MxCUniverse_getVirial(float **virial) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(virial);
    auto _virial = *univ->virial();
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCUniverse_getVirialT(struct MxParticleTypeHandle **phandles, unsigned int numTypes, float **virial) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(phandles);
    MXCPTRCHECK(virial);
    std::vector<MxParticleType*> ptypes;
    MxParticleTypeHandle *ph;
    for(unsigned int i = 0; i < numTypes; i++) { 
        ph = phandles[i];
        MXCPTRCHECK(ph); MXCPTRCHECK(ph->MxObj);
        ptypes.push_back((MxParticleType*)ph->MxObj);
    }
    auto _virial = *univ->virial(NULL, NULL, &ptypes);
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCUniverse_getVirialO(float *origin, float radius, float **virial) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(origin);
    MXCPTRCHECK(virial);
    MxVector3f _origin = MxVector3f::from(origin);
    float _radius = radius;
    auto _virial = *univ->virial(&_origin, &_radius);
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCUniverse_getVirialOT(struct MxParticleTypeHandle **phandles, unsigned int numTypes, float *origin, float radius, float **virial) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(phandles);
    MXCPTRCHECK(origin);
    MXCPTRCHECK(virial);
    std::vector<MxParticleType*> ptypes;
    MxParticleTypeHandle *ph;
    for(unsigned int i = 0; i < numTypes; i++) {
        ph = phandles[i];
        MXCPTRCHECK(ph); MXCPTRCHECK(ph->MxObj);
        ptypes.push_back((MxParticleType*)ph->MxObj);
    }
    MxVector3f _origin = MxVector3f::from(origin);
    float _radius = radius;
    auto _virial = *univ->virial(&_origin, &_radius, &ptypes);
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCUniverse_getNumParts(unsigned int *numParts) {
    MXCPTRCHECK(numParts);
    *numParts = _Engine.s.nr_parts;
    return S_OK;
}

HRESULT MxCUniverse_getParticle(unsigned int pidx, struct MxParticleHandleHandle *handle) {
    MXCPTRCHECK(handle);
    if(pidx >= _Engine.s.nr_parts) 
        return E_FAIL;
    MxParticle *p = _Engine.s.partlist[pidx];
    MXCPTRCHECK(p);
    MxParticleHandle *ph = new MxParticleHandle(p->id, p->typeId);
    handle->MxObj = (void*)ph;
    return S_OK;
}

HRESULT MxCUniverse_getCenter(float **center) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(center);
    auto c = univ->getCenter();
    MXVECTOR3_COPYFROM(c, (*center));
    return S_OK;
}

HRESULT MxCUniverse_step(double until, double dt) {
    MXUNIVERSE_STATIC_GET()
    return univ->step(until, dt);
}

HRESULT MxCUniverse_stop() {
    MXUNIVERSE_STATIC_GET()
    return univ->stop();
}

HRESULT MxCUniverse_start() {
    MXUNIVERSE_STATIC_GET()
    return univ->start();
}

HRESULT MxCUniverse_reset() {
    MXUNIVERSE_STATIC_GET()
    return univ->reset();
}

HRESULT MxCUniverse_resetSpecies() {
    MXUNIVERSE_STATIC_GET()
    univ->resetSpecies();
    return S_OK;
}

HRESULT MxCUniverse_getTemperature(double *temperature) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(temperature);
    *temperature = univ->getTemperature();
    return S_OK;
}

HRESULT MxCUniverse_getTime(double *time) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(time);
    *time = univ->getTime();
    return S_OK;
}

HRESULT MxCUniverse_getDt(double *dt) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(dt);
    *dt = univ->getDt();
    return S_OK;
}

HRESULT MxCUniverse_getBoundaryConditions(struct MxBoundaryConditionsHandle *bcs) {
    MXUNIVERSE_STATIC_GET();
    MXCPTRCHECK(bcs);
    auto _bcs = univ->getBoundaryConditions();
    MXCPTRCHECK(_bcs);
    bcs->MxObj = (void*)_bcs;
    return S_OK;
}

HRESULT MxCUniverse_getKineticEnergy(double *ke) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(ke);
    *ke = univ->getKineticEnergy();
    return S_OK;
}

HRESULT MxCUniverse_getNumTypes(int *numTypes) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(numTypes);
    *numTypes = univ->getNumTypes();
    return S_OK;
}

HRESULT MxCUniverse_getCutoff(double *cutoff) {
    MXUNIVERSE_STATIC_GET()
    MXCPTRCHECK(cutoff);
    *cutoff = univ->getCutoff();
    return S_OK;
}


////////////////////////////
// MxUniverseConfigHandle //
////////////////////////////


HRESULT MxCUniverseConfig_init(struct MxUniverseConfigHandle *handle) {
    MXCPTRCHECK(handle);
    handle->MxObj = new MxUniverseConfig();
    return S_OK;
}

HRESULT MxCUniverseConfig_destroy(struct MxUniverseConfigHandle *handle) {
    return mx::capi::destroyHandle<MxUniverseConfig, MxUniverseConfigHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCUniverseConfig_getOrigin(struct MxUniverseConfigHandle *handle, float **origin) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(origin);
    MXVECTOR3_COPYFROM(conf->origin, (*origin));
    return S_OK;
}

HRESULT MxCUniverseConfig_setOrigin(struct MxUniverseConfigHandle *handle, float *origin) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(origin);
    MXVECTOR3_COPYTO(origin, conf->origin)
    return S_OK;
}

HRESULT MxCUniverseConfig_getDim(struct MxUniverseConfigHandle *handle, float **dim) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(dim);
    MXVECTOR3_COPYFROM(conf->dim, (*dim));
    return S_OK;
}

HRESULT MxCUniverseConfig_setDim(struct MxUniverseConfigHandle *handle, float *dim) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(dim);
    MXVECTOR3_COPYTO(dim, conf->dim)
    return S_OK;
}

HRESULT MxCUniverseConfig_getCells(struct MxUniverseConfigHandle *handle, int **cells) {
    MXUNIVERSECONFIG_GET(handle);
    MXCPTRCHECK(cells);
    MXVECTOR3_COPYFROM(conf->spaceGridSize, (*cells));
    return S_OK;
}

HRESULT MxCUniverseConfig_setCells(struct MxUniverseConfigHandle *handle, int *cells) {
    MXUNIVERSECONFIG_GET(handle);
    MXCPTRCHECK(cells);
    MXVECTOR3_COPYTO(cells, conf->spaceGridSize);
    return S_OK;
}

HRESULT MxCUniverseConfig_getCutoff(struct MxUniverseConfigHandle *handle, double *cutoff) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(cutoff);
    *cutoff = conf->cutoff;
    return S_OK;
}

HRESULT MxCUniverseConfig_setCutoff(struct MxUniverseConfigHandle *handle, double cutoff) {
    MXUNIVERSECONFIG_GET(handle)
    conf->cutoff = cutoff;
    return S_OK;
}

HRESULT MxCUniverseConfig_getFlags(struct MxUniverseConfigHandle *handle, unsigned int *flags) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(flags);
    *flags = conf->flags;
    return S_OK;
}

HRESULT MxCUniverseConfig_setFlags(struct MxUniverseConfigHandle *handle, unsigned int flags) {
    MXUNIVERSECONFIG_GET(handle)
    conf->flags = flags;
    return S_OK;
}

HRESULT MxCUniverseConfig_getDt(struct MxUniverseConfigHandle *handle, double *dt) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(dt);
    *dt = conf->dt;
    return S_OK;
}

HRESULT MxCUniverseConfig_setDt(struct MxUniverseConfigHandle *handle, double dt) {
    MXUNIVERSECONFIG_GET(handle)
    conf->dt = dt;
    return S_OK;
}

HRESULT MxCUniverseConfig_getTemperature(struct MxUniverseConfigHandle *handle, double *temperature) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(temperature);
    *temperature = conf->temp;
    return S_OK;
}

HRESULT MxCUniverseConfig_setTemperature(struct MxUniverseConfigHandle *handle, double temperature) {
    MXUNIVERSECONFIG_GET(handle)
    conf->temp = temperature;
    return S_OK;
}

HRESULT MxCUniverseConfig_getNumThreads(struct MxUniverseConfigHandle *handle, unsigned int *numThreads) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(numThreads);
    *numThreads = conf->threads;
    return S_OK;
}

HRESULT MxCUniverseConfig_setNumThreads(struct MxUniverseConfigHandle *handle, unsigned int numThreads) {
    MXUNIVERSECONFIG_GET(handle)
    conf->threads = numThreads;
    return S_OK;
}

HRESULT MxCUniverseConfig_getIntegrator(struct MxUniverseConfigHandle *handle, unsigned int *integrator) {
    MXUNIVERSECONFIG_GET(handle)
    MXCPTRCHECK(integrator);
    *integrator = (unsigned int)conf->integrator;
    return S_OK;
}

HRESULT MxCUniverseConfig_setIntegrator(struct MxUniverseConfigHandle *handle, unsigned int integrator) {
    MXUNIVERSECONFIG_GET(handle)
    conf->integrator = (EngineIntegrator)integrator;
    return S_OK;
}

HRESULT MxCUniverseConfig_getBoundaryConditions(struct MxUniverseConfigHandle *handle, struct MxBoundaryConditionsArgsContainerHandle *bargsHandle) {
    MXUNIVERSECONFIG_GET(handle);
    MXCPTRCHECK(bargsHandle);
    MXCPTRCHECK(conf->boundaryConditionsPtr);
    bargsHandle->MxObj = (void*)conf->boundaryConditionsPtr;
    return S_OK;
}

HRESULT MxCUniverseConfig_setBoundaryConditions(struct MxUniverseConfigHandle *handle, struct MxBoundaryConditionsArgsContainerHandle *bargsHandle) {
    MXUNIVERSECONFIG_GET(handle);
    MXCPTRCHECK(bargsHandle); MXCPTRCHECK(bargsHandle->MxObj);
    conf->setBoundaryConditions((MxBoundaryConditionsArgsContainer*)bargsHandle->MxObj);
    return S_OK;
}
