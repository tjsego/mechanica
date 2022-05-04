/**
 * @file MxCParticle.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxParticle and associated features
 * @date 2022-03-28
 */

#include "MxCParticle.h"

#include "mechanica_c_private.h"

#include "MxCCluster.h"
#include "MxCStateVector.h"

#include <MxParticle.h>
#include <MxParticleList.hpp>
#include <MxParticleTypeList.h>
#include <state/MxSpeciesList.h>
#include <rendering/MxStyle.hpp>
#include <engine.h>


namespace mx { 

MxParticleHandle *castC(struct MxParticleHandleHandle *handle) {
    return castC<MxParticleHandle, MxParticleHandleHandle>(handle);
}

MxParticleType *castC(struct MxParticleTypeHandle *handle) {
    return castC<MxParticleType, MxParticleTypeHandle>(handle);
}

MxParticleList *castC(struct MxParticleListHandle *handle) {
    return castC<MxParticleList, MxParticleListHandle>(handle);
}

MxParticleTypeList *castC(struct MxParticleTypeListHandle *handle) {
    return castC<MxParticleTypeList, MxParticleTypeListHandle>(handle);
}

}

#define MXPARTICLEHANDLE_GET(handle) \
    MxParticleHandle *phandle = mx::castC<MxParticleHandle, MxParticleHandleHandle>(handle); \
    MXCPTRCHECK(phandle);

#define MXPARTICLEHANDLE_GETN(handle, name) \
    MxParticleHandle *name = mx::castC<MxParticleHandle, MxParticleHandleHandle>(handle); \
    MXCPTRCHECK(name);

#define MXPTYPEHANDLE_GET(handle) \
    MxParticleType *ptype = mx::castC<MxParticleType, MxParticleTypeHandle>(handle); \
    MXCPTRCHECK(ptype);

#define MXPARTICLELIST_GET(handle) \
    MxParticleList *plist = mx::castC<MxParticleList, MxParticleListHandle>(handle); \
    MXCPTRCHECK(plist);

#define MXPARTICLETYPELIST_GET(handle) \
    MxParticleTypeList *ptlist = mx::castC<MxParticleTypeList, MxParticleTypeListHandle>(handle); \
    MXCPTRCHECK(ptlist);


/////////////////////
// MxCParticleType //
/////////////////////


struct MxCParticleType MxCParticleTypeDef_init() {
    struct MxCParticleType MxCParticleTypeDef = {
        1.0, 
        0.0, 
        1.0, 
        NULL, 
        0.0, 
        0.0, 
        0.0, 
        0, 
        0, 
        NULL, 
        NULL, 
        NULL, 
        0, 
        NULL
    };
    return MxCParticleTypeDef;
}


//////////////////////////
// MxCParticleTypeStyle //
//////////////////////////


struct MxCParticleTypeStyle MxCParticleTypeStyleDef_init() {
    struct MxCParticleTypeStyle MxCParticleTypeStyleDef = {
        NULL, 
        1, 
        NULL, 
        "rainbow", 
        0.0, 
        1.0
    };
    return MxCParticleTypeStyleDef;
}


////////////////////////
// MxParticleDynamics //
////////////////////////


HRESULT MxCParticleDynamics_init(struct MxParticleDynamicsEnumHandle *handle) {
    MXCPTRCHECK(handle);
    handle->PARTICLE_NEWTONIAN = PARTICLE_NEWTONIAN;
    handle->PARTICLE_OVERDAMPED = PARTICLE_OVERDAMPED;
    return S_OK;
}


/////////////////////
// MxParticleFlags //
/////////////////////


HRESULT MxCParticleFlags_init(struct MxParticleFlagsHandle *handle) {
    MXCPTRCHECK(handle);
    handle->PARTICLE_NONE = PARTICLE_NONE;
    handle->PARTICLE_GHOST = PARTICLE_GHOST;
    handle->PARTICLE_CLUSTER = PARTICLE_CLUSTER;
    handle->PARTICLE_BOUND = PARTICLE_BOUND;
    handle->PARTICLE_FROZEN_X = PARTICLE_FROZEN_X;
    handle->PARTICLE_FROZEN_Y = PARTICLE_FROZEN_Y;
    handle->PARTICLE_FROZEN_Z = PARTICLE_FROZEN_Z;
    handle->PARTICLE_FROZEN = PARTICLE_FROZEN;
    handle->PARTICLE_LARGE = PARTICLE_LARGE;
    return S_OK;
}


//////////////////////
// MxParticleHandle //
//////////////////////


HRESULT MxCParticleHandle_init(struct MxParticleHandleHandle *handle, unsigned int pid) {
    MXCPTRCHECK(handle);
    if(pid >= _Engine.s.size_parts) 
        return E_FAIL;
    for(unsigned int i = 0; i < _Engine.s.size_parts; i++) {
        MxParticle *p = _Engine.s.partlist[i];
        if(p && p->id == pid) {
            MxParticleHandle *ph = new MxParticleHandle(p->id, p->typeId);
            handle->MxObj = (void*)ph;
            return S_OK;
        }
    }
    return E_FAIL;
}

HRESULT MxCParticleHandle_destroy(struct MxParticleHandleHandle *handle) {
    return mx::capi::destroyHandle<MxParticleHandle, MxParticleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCParticleHandle_getType(struct MxParticleHandleHandle *handle, struct MxParticleTypeHandle *typeHandle) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(typeHandle);
    MxParticleType *ptype = phandle->type();
    MXCPTRCHECK(ptype);

    typeHandle->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCParticleHandle_split(struct MxParticleHandleHandle *handle, struct MxParticleHandleHandle *newParticleHandle) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(newParticleHandle);

    auto nphandle = phandle->fission();
    MXCPTRCHECK(nphandle);

    newParticleHandle->MxObj = (void*)nphandle;

    return S_OK;
}

HRESULT MxCParticleHandle_destroyParticle(struct MxParticleHandleHandle *handle) {
    MXPARTICLEHANDLE_GET(handle);
    return phandle->destroy();
}

HRESULT MxCParticleHandle_sphericalPosition(struct MxParticleHandleHandle *handle, float **position) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(position);
    auto sp = phandle->sphericalPosition();
    MXVECTOR3_COPYFROM(sp, (*position));
    return S_OK;
}

HRESULT MxCParticleHandle_sphericalPositionPoint(struct MxParticleHandleHandle *handle, float *origin, float **position) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(origin);
    MXCPTRCHECK(position);
    MxVector3f _origin = MxVector3f::from(origin);
    auto sp = phandle->sphericalPosition(0, &_origin);
    MXVECTOR3_COPYFROM(sp, (*position));
    return S_OK;
}

HRESULT MxCParticleHandle_relativePosition(struct MxParticleHandleHandle *handle, float *origin, float **position) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(origin);
    MXCPTRCHECK(position);
    auto p = phandle->relativePosition(MxVector3f::from(origin));
    MXVECTOR3_COPYFROM(p, (*position));
    return S_OK;
}

HRESULT MxCParticleHandle_become(struct MxParticleHandleHandle *handle, struct MxParticleTypeHandle *typeHandle) {
    MXPARTICLEHANDLE_GET(handle);
    MXPTYPEHANDLE_GET(typeHandle);
    return phandle->become(ptype);
}

HRESULT package_parts(MxParticleList *plist, struct MxParticleHandleHandle **hlist) {
    MXCPTRCHECK(plist);
    MXCPTRCHECK(hlist);
    MxParticleHandleHandle *_hlist = (MxParticleHandleHandle*)malloc(plist->nr_parts * sizeof(MxParticleHandleHandle));
    if(!_hlist) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < plist->nr_parts; i++) {
        MxParticle *p = _Engine.s.partlist[plist->parts[i]];
        MXCPTRCHECK(p);
        MxParticleHandle *ph = new MxParticleHandle(p->id, p->typeId);
        _hlist[i].MxObj = (void*)ph;
    }
    *hlist = _hlist;
    return S_OK;
}

HRESULT MxCParticleHandle_neighborsD(struct MxParticleHandleHandle *handle, float distance, struct MxParticleHandleHandle **neighbors, int *numNeighbors) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(neighbors);
    MXCPTRCHECK(numNeighbors);
    float _distance = distance;
    auto nbs = phandle->neighbors(&_distance);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT MxCParticleHandle_neighborsT(struct MxParticleHandleHandle *handle, 
                                     struct MxParticleTypeHandle *ptypes, 
                                     int numTypes, 
                                     struct MxParticleHandleHandle **neighbors, 
                                     int *numNeighbors) 
{
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(ptypes);
    MXCPTRCHECK(neighbors);
    MXCPTRCHECK(numNeighbors);
    std::vector<MxParticleType> _ptypes;
    for(unsigned int i = 0; i < numTypes; i++) {
        MxParticleTypeHandle pth = ptypes[i];
        MXCPTRCHECK(pth.MxObj);
        _ptypes.push_back(*(MxParticleType*)pth.MxObj);
    }
    auto nbs = phandle->neighbors(0, &_ptypes);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT MxCParticleHandle_neighborsDT(struct MxParticleHandleHandle *handle, 
                                      float distance, 
                                      struct MxParticleTypeHandle *ptypes, 
                                      int numTypes, 
                                      struct MxParticleHandleHandle **neighbors, 
                                      int *numNeighbors) 
{
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(ptypes);
    MXCPTRCHECK(neighbors);
    MXCPTRCHECK(numNeighbors);
    std::vector<MxParticleType> _ptypes;
    for(unsigned int i = 0; i < numTypes; i++) {
        MxParticleTypeHandle pth = ptypes[i];
        MXCPTRCHECK(pth.MxObj);
        _ptypes.push_back(*(MxParticleType*)pth.MxObj);
    }
    float _distance = distance;
    auto nbs = phandle->neighbors(&_distance, &_ptypes);
    *numNeighbors = nbs->nr_parts;
    return package_parts(nbs, neighbors);
}

HRESULT MxCParticleHandle_getBondedNeighbors(struct MxParticleHandleHandle *handle, struct MxParticleHandleHandle **neighbors, int *numNeighbors) 
{
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(neighbors);
    MXCPTRCHECK(numNeighbors);
    auto plist = phandle->getBondedNeighbors();
    *numNeighbors = plist->nr_parts;
    if(plist->nr_parts == 0) 
        return S_OK;
    return package_parts(plist, neighbors);
}

HRESULT MxCParticleHandle_distance(struct MxParticleHandleHandle *handle, struct MxParticleHandleHandle *other, float *distance) {
    MXPARTICLEHANDLE_GET(handle);
    MXPARTICLEHANDLE_GETN(other, ohandle);
    MXCPTRCHECK(distance);
    *distance = phandle->distance(ohandle);
    return S_OK;
}

HRESULT MxCParticleHandle_getMass(struct MxParticleHandleHandle *handle, double *mass) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(mass);
    *mass = phandle->getMass();
    return S_OK;
}

HRESULT MxCParticleHandle_setMass(struct MxParticleHandleHandle *handle, double mass) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setMass(mass);
    return S_OK;
}

HRESULT MxCParticleHandle_getFrozen(struct MxParticleHandleHandle *handle, bool *frozen) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = phandle->getFrozen();
    return S_OK;
}

HRESULT MxCParticleHandle_setFrozen(struct MxParticleHandleHandle *handle, bool frozen) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setFrozen(frozen);
    return S_OK;
}

HRESULT MxCParticleHandle_getFrozenX(struct MxParticleHandleHandle *handle, bool *frozen) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = phandle->getFrozenX();
    return S_OK;
}

HRESULT MxCParticleHandle_setFrozenX(struct MxParticleHandleHandle *handle, bool frozen) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setFrozenX(frozen);
    return S_OK;
}

HRESULT MxCParticleHandle_getFrozenY(struct MxParticleHandleHandle *handle, bool *frozen) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = phandle->getFrozenY();
    return S_OK;
}

HRESULT MxCParticleHandle_setFrozenY(struct MxParticleHandleHandle *handle, bool frozen) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setFrozenY(frozen);
    return S_OK;
}

HRESULT MxCParticleHandle_getFrozenZ(struct MxParticleHandleHandle *handle, bool *frozen) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = phandle->getFrozenZ();
    return S_OK;
}

HRESULT MxCParticleHandle_setFrozenZ(struct MxParticleHandleHandle *handle, bool frozen) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setFrozenZ(frozen);
    return S_OK;
}

HRESULT MxCParticleHandle_getStyle(struct MxParticleHandleHandle *handle, struct MxStyleHandle *style) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(style);
    MxStyle *pstyle = phandle->getStyle();
    MXCPTRCHECK(pstyle);
    style->MxObj = (void*)pstyle;
    return S_OK;
}

HRESULT MxCParticleHandle_hasStyle(struct MxParticleHandleHandle *handle, bool *hasStyle) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(hasStyle);
    *hasStyle = phandle->getStyle() != NULL;
    return S_OK;
}

HRESULT MxCParticleHandle_setStyle(struct MxParticleHandleHandle *handle, struct MxStyleHandle *style) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(style); MXCPTRCHECK(style->MxObj);
    phandle->setStyle((MxStyle*)style->MxObj);
    return S_OK;
}

HRESULT MxCParticleHandle_getAge(struct MxParticleHandleHandle *handle, double *age) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(age);
    *age = phandle->getAge();
    return S_OK;
}

HRESULT MxCParticleHandle_getRadius(struct MxParticleHandleHandle *handle, double *radius) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(radius);
    *radius = phandle->getRadius();
    return S_OK;
}

HRESULT MxCParticleHandle_setRadius(struct MxParticleHandleHandle *handle, double radius) {
    MXPARTICLEHANDLE_GET(handle);
    phandle->setRadius(radius);
    return S_OK;
}

HRESULT MxCParticleHandle_getName(struct MxParticleHandleHandle *handle, char **name, unsigned int *numChars) {
    MXPARTICLEHANDLE_GET(handle);
    return mx::capi::str2Char(phandle->getName(), name, numChars);
}

HRESULT MxCParticleHandle_getName2(struct MxParticleHandleHandle *handle, char **name, unsigned int *numChars) {
    MXPARTICLEHANDLE_GET(handle);
    return mx::capi::str2Char(phandle->getName2(), name, numChars);
}

HRESULT MxCParticleHandle_getPosition(struct MxParticleHandleHandle *handle, float **position) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(position);
    auto p = phandle->getPosition();
    MXVECTOR3_COPYFROM(p, (*position));
    return S_OK;
}

HRESULT MxCParticleHandle_setPosition(struct MxParticleHandleHandle *handle, float *position) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(position);
    phandle->setPosition(MxVector3f::from(position));
    return S_OK;
}

HRESULT MxCParticleHandle_getVelocity(struct MxParticleHandleHandle *handle, float **velocity) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(velocity);
    auto v = phandle->getVelocity();
    MXVECTOR3_COPYFROM(v, (*velocity));
    return S_OK;
}

HRESULT MxCParticleHandle_setVelocity(struct MxParticleHandleHandle *handle, float *velocity) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(velocity);
    phandle->setVelocity(MxVector3f::from(velocity));
    return S_OK;
}

HRESULT MxCParticleHandle_getForce(struct MxParticleHandleHandle *handle, float **force) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(force);
    auto f = phandle->getForce();
    MXVECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT MxCParticleHandle_getForceInit(struct MxParticleHandleHandle *handle, float **force) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(force);
    auto f = phandle->getForceInit();
    MXVECTOR3_COPYFROM(f, (*force));
    return S_OK;
}

HRESULT MxCParticleHandle_setForceInit(struct MxParticleHandleHandle *handle, float *force) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(force);
    phandle->setForceInit(MxVector3f::from(force));
    return S_OK;
}

HRESULT MxCParticleHandle_getId(struct MxParticleHandleHandle *handle, int *pid) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(pid);
    *pid = phandle->getId();
    return S_OK;
}

HRESULT MxCParticleHandle_getTypeId(struct MxParticleHandleHandle *handle, int *tid) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(tid);
    *tid = phandle->getTypeId();
    return S_OK;
}

HRESULT MxCParticleHandle_getClusterId(struct MxParticleHandleHandle *handle, int *cid) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(cid);
    *cid = phandle->getClusterId();
    return S_OK;
}

HRESULT MxCParticleHandle_getFlags(struct MxParticleHandleHandle *handle, int *flags) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(flags);
    *flags = phandle->getFlags();
    return S_OK;
}

HRESULT MxCParticleHandle_hasSpecies(struct MxParticleHandleHandle *handle, bool *flag) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(flag);
    *flag = phandle->getSpecies() != NULL;
    return S_OK;
}

HRESULT MxCParticleHandle_getSpecies(struct MxParticleHandleHandle *handle, struct MxStateVectorHandle *svec) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(svec);
    auto _svec = phandle->getSpecies();
    MXCPTRCHECK(_svec);
    svec->MxObj = (void*)_svec;
    return S_OK;
}

HRESULT MxCParticleHandle_setSpecies(struct MxParticleHandleHandle *handle, struct MxStateVectorHandle *svec) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(svec); MXCPTRCHECK(svec->MxObj);
    MxParticle *p = phandle->part();
    MXCPTRCHECK(p);
    p->state_vector = (MxStateVector*)svec->MxObj;
    return S_OK;
}

HRESULT MxCParticleHandle_toCluster(struct MxParticleHandleHandle *handle, struct MxClusterParticleHandleHandle *chandle) {
    MXPARTICLEHANDLE_GET(handle);
    MXCPTRCHECK(chandle);
    if(phandle->getClusterId() < 0) 
        return E_FAIL;
    chandle->MxObj = (void*)phandle;
    return S_OK;
}

HRESULT MxCParticleHandle_toString(struct MxParticleHandleHandle *handle, char **str, unsigned int *numChars) {
    MXPARTICLEHANDLE_GET(handle);
    auto p = phandle->part();
    MXCPTRCHECK(p);
    return mx::capi::str2Char(p->toString(), str, numChars);
}


////////////////////
// MxParticleType //
////////////////////


HRESULT MxCParticleType_init(struct MxParticleTypeHandle *handle) {
    MXCPTRCHECK(handle);
    MxParticleType *ptype = new MxParticleType(true);
    handle->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCParticleType_initD(struct MxParticleTypeHandle *handle, struct MxCParticleType pdef) {
    HRESULT result = MxCParticleType_init(handle);
    if(result != S_OK) 
        return result;
    MXPTYPEHANDLE_GET(handle);

    ptype->mass = pdef.mass;
    ptype->radius = pdef.radius;
    if(pdef.target_energy) ptype->target_energy = *pdef.target_energy;
    ptype->minimum_radius = pdef.minimum_radius;
    ptype->dynamics = pdef.dynamics;
    ptype->setFrozen(pdef.frozen);
    if(pdef.name) 
        if((result = MxCParticleType_setName(handle, pdef.name)) != S_OK) 
            return result;
    if(pdef.numSpecies > 0) {
        ptype->species = new MxSpeciesList();
        for(unsigned int i = 0; i < pdef.numSpecies; i++) 
            if((result = ptype->species->insert(pdef.species[i])) != S_OK) 
                return result;
    }
    if(pdef.style) {
        if(pdef.style->color) 
            ptype->style->setColor(pdef.style->color);
        else if(pdef.style->speciesName) 
            ptype->style->newColorMapper(ptype, pdef.style->speciesName, pdef.style->speciesMapName, pdef.style->speciesMapMin, pdef.style->speciesMapMax);
        ptype->style->setVisible(pdef.style->visible);
    }

    return S_OK;
}

HRESULT MxCParticleType_getName(struct MxParticleTypeHandle *handle, char **name, unsigned int *numChars) {
    MXPTYPEHANDLE_GET(handle);
    return mx::capi::str2Char(std::string(ptype->name), name, numChars);
}

HRESULT MxCParticleType_setName(struct MxParticleTypeHandle *handle, const char *name) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(name);
    std::string ns(name);
    std::strncpy(ptype->name, ns.c_str(), ns.size());
    return S_OK;
}

HRESULT MxCParticleType_getId(struct MxParticleTypeHandle *handle, int *id) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(id);
    *id = ptype->id;
    return S_OK;
}

HRESULT MxCParticleType_getTypeFlags(struct MxParticleTypeHandle *handle, unsigned int *flags) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(flags);
    *flags = ptype->type_flags;
    return S_OK;
}

HRESULT MxCParticleType_setTypeFlags(struct MxParticleTypeHandle *handle, unsigned int flags) {
    MXPTYPEHANDLE_GET(handle);
    ptype->type_flags = flags;
    return S_OK;
}

HRESULT MxCParticleType_getParticleFlags(struct MxParticleTypeHandle *handle, unsigned int *flags) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(flags);
    *flags = ptype->particle_flags;
    return S_OK;
}

HRESULT MxCParticleType_setParticleFlags(struct MxParticleTypeHandle *handle, unsigned int flags) {
    MXPTYPEHANDLE_GET(handle);
    ptype->particle_flags = flags;
    return S_OK;
}

HRESULT MxCParticleType_getStyle(struct MxParticleTypeHandle *handle, struct MxStyleHandle *style) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(style);
    MXCPTRCHECK(ptype->style);
    style->MxObj = (void*)ptype->style;
    return S_OK;
}

HRESULT MxCParticleType_setStyle(struct MxParticleTypeHandle *handle, struct MxStyleHandle *style) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(style); MXCPTRCHECK(style->MxObj);
    ptype->style = (MxStyle*)style->MxObj;
    return S_OK;
}

HRESULT MxCParticleType_hasSpecies(struct MxParticleTypeHandle *handle, bool *flag) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(flag);
    *flag = ptype->species != NULL;
    return S_OK;
}

HRESULT MxCParticleType_getSpecies(struct MxParticleTypeHandle *handle, struct MxSpeciesListHandle *slist) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(ptype->species)
    MXCPTRCHECK(slist);
    slist->MxObj = (void*)ptype->species;
    return S_OK;
}

HRESULT MxCParticleType_setSpecies(struct MxParticleTypeHandle *handle, struct MxSpeciesListHandle *slist) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(slist); MXCPTRCHECK(slist->MxObj);
    ptype->species = (MxSpeciesList*)slist->MxObj;
    return S_OK;
}

HRESULT MxCParticleType_getMass(struct MxParticleTypeHandle *handle, double *mass) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(mass);
    *mass = ptype->mass;
    return S_OK;
}

HRESULT MxCParticleType_setMass(struct MxParticleTypeHandle *handle, double mass) {
    MXPTYPEHANDLE_GET(handle);
    ptype->mass = mass;
    return S_OK;
}

HRESULT MxCParticleType_getRadius(struct MxParticleTypeHandle *handle, double *radius) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(radius);
    *radius = ptype->radius;
    return S_OK;
}

HRESULT MxCParticleType_setRadius(struct MxParticleTypeHandle *handle, double radius) {
    MXPTYPEHANDLE_GET(handle);
    ptype->radius = radius;
    return S_OK;
}

HRESULT MxCParticleType_getKineticEnergy(struct MxParticleTypeHandle *handle, double *kinetic_energy) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(kinetic_energy);
    *kinetic_energy = ptype->kinetic_energy;
    return S_OK;
}

HRESULT MxCParticleType_getPotentialEnergy(struct MxParticleTypeHandle *handle, double *potential_energy) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(potential_energy);
    *potential_energy = ptype->potential_energy;
    return S_OK;
}

HRESULT MxCParticleType_getTargetEnergy(struct MxParticleTypeHandle *handle, double *target_energy) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(target_energy);
    *target_energy = ptype->target_energy;
    return S_OK;
}

HRESULT MxCParticleType_setTargetEnergy(struct MxParticleTypeHandle *handle, double target_energy) {
    MXPTYPEHANDLE_GET(handle);
    ptype->target_energy = target_energy;
    return S_OK;
}

HRESULT MxCParticleType_getMinimumRadius(struct MxParticleTypeHandle *handle, double *minimum_radius) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(minimum_radius);
    *minimum_radius = ptype->minimum_radius;
    return S_OK;
}

HRESULT MxCParticleType_setMinimumRadius(struct MxParticleTypeHandle *handle, double minimum_radius) {
    MXPTYPEHANDLE_GET(handle);
    ptype->minimum_radius = minimum_radius;
    return S_OK;
}

HRESULT MxCParticleType_getDynamics(struct MxParticleTypeHandle *handle, unsigned char *dynamics) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(dynamics);
    *dynamics = ptype->dynamics;
    return S_OK;
}

HRESULT MxCParticleType_setDynamics(struct MxParticleTypeHandle *handle, unsigned char dynamics) {
    MXPTYPEHANDLE_GET(handle);
    ptype->dynamics = dynamics;
    return S_OK;
}

HRESULT MxCParticleType_getNumParticles(struct MxParticleTypeHandle *handle, int *numParts) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(numParts);
    *numParts = ptype->parts.nr_parts;
    return S_OK;
}

HRESULT MxCParticleType_getParticle(struct MxParticleTypeHandle *handle, int i, struct MxParticleHandleHandle *phandle) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(phandle);
    MxParticle *p = ptype->particle(i);
    MXCPTRCHECK(p);
    
    MxParticleHandle *ph = new MxParticleHandle(p->id, ptype->id);
    phandle->MxObj = (void*)ph;
    return S_OK;
}

HRESULT MxCParticleType_isCluster(struct MxParticleTypeHandle *handle, bool *isCluster) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(isCluster);
    *isCluster = ptype->isCluster();
    return S_OK;
}

HRESULT MxCParticleType_toCluster(struct MxParticleTypeHandle *handle, struct MxClusterParticleTypeHandle *chandle) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(chandle);
    if(!ptype->isCluster()) 
        return E_FAIL;
    chandle->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCParticleType_createParticle(struct MxParticleTypeHandle *handle, int *pid, float *position, float *velocity) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(pid);
    MxVector3f p, v, *p_ptr = NULL, *v_ptr = NULL;
    if(position) {
        p = MxVector3f::from(position);
        p_ptr = &p;
    }
    if(velocity) {
        v = MxVector3f::from(velocity);
        v_ptr = &v;
    }
    MxParticleHandle *ph = (*ptype)(p_ptr, v_ptr);
    if(!ph) 
        return E_FAIL;

    *pid = ph->id;
    return S_OK;
}

HRESULT MxCParticleType_createParticleS(struct MxParticleTypeHandle *handle, int *pid, const char *str) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(pid);
    MXCPTRCHECK(str);
    MxParticleHandle *ph = (*ptype)(str);
    *pid = ph->id;
    return S_OK;
}

HRESULT MxCParticleType_factory(struct MxParticleTypeHandle *handle, int **pids, unsigned int nr_parts, float *positions, float *velocities) {
    MXPTYPEHANDLE_GET(handle);
    if(nr_parts == 0) 
        return E_FAIL;

    unsigned int nr_parts_ui = (unsigned int)nr_parts;

    std::vector<MxVector3f> _positions, *_positions_p = NULL;
    std::vector<MxVector3f> _velocities, *_velocities_p = NULL;

    if(positions) {
        _positions.reserve(nr_parts);
        for(unsigned int i = 0; i < nr_parts; i++) 
            _positions.push_back(MxVector3f::from(&positions[3 * i]));
        _positions_p = &_positions;
    }
    if(velocities) {
        _velocities.reserve(nr_parts);
        for(unsigned int i = 0; i < nr_parts; i++) 
            _velocities.push_back(MxVector3f::from(&velocities[3 * i]));
        _velocities_p = &_velocities;
    }

    std::vector<int> _pids_v = ptype->factory(nr_parts_ui, _positions_p, _velocities_p);
    if(pids) 
        std::copy(_pids_v.begin(), _pids_v.end(), *pids);

    return S_OK;
}

HRESULT MxCParticleType_newType(struct MxParticleTypeHandle *handle, const char *_name, struct MxParticleTypeHandle *newTypehandle) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(_name);
    MXCPTRCHECK(newTypehandle);
    MxParticleType *ptypeNew = ptype->newType(_name);
    MXCPTRCHECK(ptypeNew);
    newTypehandle->MxObj = (void*)ptypeNew;
    return S_OK;
}

HRESULT MxCParticleType_registerType(struct MxParticleTypeHandle *handle) {
    MXPTYPEHANDLE_GET(handle);
    HRESULT result = ptype->registerType();
    if(result != S_OK) 
        return result;
    auto pid = ptype->id;
    handle->MxObj = (void*)&_Engine.types[pid];
    return S_OK;
}

HRESULT MxCParticleType_isRegistered(struct MxParticleTypeHandle *handle, bool *isRegistered) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(isRegistered);
    *isRegistered = ptype->isRegistered();
    return S_OK;
}

HRESULT MxCParticleType_getFrozen(struct MxParticleTypeHandle *handle, bool *frozen) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = ptype->getFrozen();
    return S_OK;
}

HRESULT MxCParticleType_setFrozen(struct MxParticleTypeHandle *handle, bool frozen) {
    MXPTYPEHANDLE_GET(handle);
    ptype->setFrozen(frozen);
    return S_OK;
}

HRESULT MxCParticleType_getFrozenX(struct MxParticleTypeHandle *handle, bool *frozen) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = ptype->getFrozenX();
    return S_OK;
}

HRESULT MxCParticleType_setFrozenX(struct MxParticleTypeHandle *handle, bool frozen) {
    MXPTYPEHANDLE_GET(handle);
    ptype->setFrozenX(frozen);
    return S_OK;
}

HRESULT MxCParticleType_getFrozenY(struct MxParticleTypeHandle *handle, bool *frozen) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = ptype->getFrozenY();
    return S_OK;
}

HRESULT MxCParticleType_setFrozenY(struct MxParticleTypeHandle *handle, bool frozen) {
    MXPTYPEHANDLE_GET(handle);
    ptype->setFrozenY(frozen);
    return S_OK;
}

HRESULT MxCParticleType_getFrozenZ(struct MxParticleTypeHandle *handle, bool *frozen) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(frozen);
    *frozen = ptype->getFrozenZ();
    return S_OK;
}

HRESULT MxCParticleType_setFrozenZ(struct MxParticleTypeHandle *handle, bool frozen) {
    MXPTYPEHANDLE_GET(handle);
    ptype->setFrozenZ(frozen);
    return S_OK;
}

HRESULT MxCParticleType_getTemperature(struct MxParticleTypeHandle *handle, double *temperature) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(temperature);
    *temperature = ptype->getTemperature();
    return S_OK;
}

HRESULT MxCParticleType_getTargetTemperature(struct MxParticleTypeHandle *handle, double *temperature) {
    MXPTYPEHANDLE_GET(handle);
    MXCPTRCHECK(temperature);
    *temperature = ptype->getTargetTemperature();
    return S_OK;
}

HRESULT MxCParticleType_setTargetTemperature(struct MxParticleTypeHandle *handle, double temperature) {
    MXPTYPEHANDLE_GET(handle);
    ptype->setTargetTemperature(temperature);
    return S_OK;
}

HRESULT MxCParticleType_toString(struct MxParticleTypeHandle *handle, char **str, unsigned int *numChars) {
    MXPTYPEHANDLE_GET(handle);
    return mx::capi::str2Char(ptype->toString(), str, numChars);
}

HRESULT MxCParticleType_fromString(struct MxParticleTypeHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxParticleType *ptype = MxParticleType::fromString(str);
    MXCPTRCHECK(ptype);
    handle->MxObj = (void*)ptype;
    return S_OK;
}


////////////////////
// MxParticleList //
////////////////////


HRESULT MxCParticleList_init(struct MxParticleListHandle *handle) {
    MXCPTRCHECK(handle);
    MxParticleList *plist = new MxParticleList();
    handle->MxObj = (void*)plist;
    return S_OK;
}

HRESULT MxCParticleList_initP(struct MxParticleListHandle *handle, struct MxParticleHandleHandle **particles, unsigned int numParts) {
    MXCPTRCHECK(handle);
    std::vector<MxParticleHandle*> _particles;
    for(unsigned int i = 0; i < numParts; i++) {
        MxParticleHandleHandle *phandle = particles[i];
        MXCPTRCHECK(phandle); MXCPTRCHECK(phandle->MxObj);
        _particles.push_back((MxParticleHandle*)phandle->MxObj);
    }

    MxParticleList *plist = new MxParticleList(_particles);
    handle->MxObj = (void*)plist;
    return S_OK;
}

HRESULT MxCParticleList_initI(struct MxParticleListHandle *handle, int *parts, unsigned int numParts) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(parts);
    MxParticleList *plist = new MxParticleList(numParts, parts);
    handle->MxObj = (void*)plist;
    return S_OK;
}

HRESULT MxCParticleList_copy(struct MxParticleListHandle *source, struct MxParticleListHandle *destination) {
    MXPARTICLELIST_GET(source);
    MXCPTRCHECK(destination);
    MxParticleList *_destination = new MxParticleList(*plist);
    destination->MxObj = (void*)_destination;
    return S_OK;
}

HRESULT MxCParticleList_destroy(struct MxParticleListHandle *handle) {
    return mx::capi::destroyHandle<MxParticleList, MxParticleListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCParticleList_getIds(struct MxParticleListHandle *handle, int **parts) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(parts);
    *parts = plist->parts;
    return S_OK;
}

HRESULT MxCParticleList_getNumParts(struct MxParticleListHandle *handle, unsigned int *numParts) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(numParts);
    *numParts = plist->nr_parts;
    return S_OK;
}

HRESULT MxCParticleList_free(struct MxParticleListHandle *handle) {
    MXPARTICLELIST_GET(handle);
    plist->free();
    return S_OK;
}

HRESULT MxCParticleList_insertI(struct MxParticleListHandle *handle, int item, unsigned int *index) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(index);
    *index = plist->insert(item);
    return S_OK;
}

HRESULT MxCParticleList_insertP(struct MxParticleListHandle *handle, struct MxParticleHandleHandle *particle, unsigned int *index) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(particle); MXCPTRCHECK(particle->MxObj);
    MXCPTRCHECK(index);
    *index = plist->insert((MxParticleHandle*)particle->MxObj);
    return S_OK;
}

HRESULT MxCParticleList_remove(struct MxParticleListHandle *handle, int id) {
    MXPARTICLELIST_GET(handle);
    plist->remove(id);
    return S_OK;
}

HRESULT MxCParticleList_extend(struct MxParticleListHandle *handle, struct MxParticleListHandle *other) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(other); MXCPTRCHECK(other->MxObj);
    plist->extend(*(MxParticleList*)other->MxObj);
    return S_OK;
}

HRESULT MxCParticleList_item(struct MxParticleListHandle *handle, unsigned int i, struct MxParticleHandleHandle *item) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(item);
    auto *part = plist->item(i);
    MXCPTRCHECK(part);
    item->MxObj = (void*)part;
    return E_FAIL;
}

HRESULT MxCParticleList_getAll(struct MxParticleListHandle *handle) {
    MXCPTRCHECK(handle);
    MxParticleList *plist = MxParticleList::all();
    handle->MxObj = (void*)plist;
    return S_OK;
}

HRESULT MxCParticleList_getVirial(struct MxParticleListHandle *handle, float **virial) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(virial);
    auto _virial = plist->getVirial();
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCParticleList_getRadiusOfGyration(struct MxParticleListHandle *handle, float *rog) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(rog);
    *rog = plist->getRadiusOfGyration();
    return S_OK;
}

HRESULT MxCParticleList_getCenterOfMass(struct MxParticleListHandle *handle, float **com) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(com);
    auto _com = plist->getCenterOfMass();
    MXVECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT MxCParticleList_getCentroid(struct MxParticleListHandle *handle, float **cent) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(cent);
    auto _cent = plist->getCentroid();
    MXVECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT MxCParticleList_getMomentOfInertia(struct MxParticleListHandle *handle, float **moi) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(moi);
    auto _moi = plist->getMomentOfInertia();
    MXMATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}

HRESULT MxCParticleList_getPositions(struct MxParticleListHandle *handle, float **positions) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(positions);
    return mx::capi::copyVecVecs3_2Arr(plist->getPositions(), positions);
}

HRESULT MxCParticleList_getVelocities(struct MxParticleListHandle *handle, float **velocities) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(velocities);
    return mx::capi::copyVecVecs3_2Arr(plist->getVelocities(), velocities);
}

HRESULT MxCParticleList_getForces(struct MxParticleListHandle *handle, float **forces) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(forces);
    return mx::capi::copyVecVecs3_2Arr(plist->getForces(), forces);
}

HRESULT MxCParticleList_sphericalPositions(struct MxParticleListHandle *handle, float **coordinates) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(coordinates)
    return mx::capi::copyVecVecs3_2Arr(plist->sphericalPositions(), coordinates);
}

HRESULT MxCParticleList_sphericalPositionsO(struct MxParticleListHandle *handle, float *origin, float **coordinates) {
    MXPARTICLELIST_GET(handle);
    MXCPTRCHECK(origin);
    MXCPTRCHECK(coordinates);
    MxVector3f _origin = MxVector3f::from(origin);
    return mx::capi::copyVecVecs3_2Arr(plist->sphericalPositions(&_origin), coordinates);
}

HRESULT MxCParticleList_toString(struct MxParticleListHandle *handle, char **str, unsigned int *numChars) {
    MXPARTICLELIST_GET(handle);
    return mx::capi::str2Char(plist->toString(), str, numChars);
}

HRESULT MxCParticleList_fromString(struct MxParticleListHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    handle->MxObj = (void*)MxParticleList::fromString(str);
    return S_OK;
}


////////////////////////
// MxParticleTypeList //
////////////////////////


HRESULT MxCParticleTypeList_init(struct MxParticleTypeListHandle *handle) {
    MXCPTRCHECK(handle);
    MxParticleTypeList *ptlist = new MxParticleTypeList();
    handle->MxObj = (void*)ptlist;
    return S_OK;
}

HRESULT MxCParticleTypeList_initP(struct MxParticleTypeListHandle *handle, struct MxParticleTypeHandle **parts, unsigned int numParts) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(parts);
    std::vector<MxParticleType*> _parts;
    for(unsigned int i = 0; i < numParts; i++) {
        MxParticleTypeHandle *phandle = parts[i];
        MXCPTRCHECK(phandle); MXCPTRCHECK(phandle->MxObj);
        _parts.push_back((MxParticleType*)phandle->MxObj);
    }
    MxParticleTypeList *plist = new MxParticleTypeList(_parts);
    handle->MxObj = (void*)plist;
    return S_OK;
}

HRESULT MxCParticleTypeList_initI(struct MxParticleTypeListHandle *handle, int *parts, unsigned int numParts) {
    MXCPTRCHECK(handle);
    MxParticleTypeList *ptlist = new MxParticleTypeList(numParts, parts);
    handle->MxObj = (void*)ptlist;
    return S_OK;
}

HRESULT MxCParticleTypeList_copy(struct MxParticleTypeListHandle *source, struct MxParticleTypeListHandle *destination) {
    MXPARTICLETYPELIST_GET(source);
    MXCPTRCHECK(destination);
    MxParticleTypeList *_destination = new MxParticleTypeList(*ptlist);
    destination->MxObj = (void*)_destination;
    return S_OK;
}

HRESULT MxCParticleTypeList_destroy(struct MxParticleTypeListHandle *handle) {
    return mx::capi::destroyHandle<MxParticleTypeList, MxParticleTypeListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCParticleTypeList_getIds(struct MxParticleTypeListHandle *handle, int **parts) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(parts);
    *parts = ptlist->parts;
    return S_OK;
}

HRESULT MxCParticleTypeList_getNumParts(struct MxParticleTypeListHandle *handle, unsigned int *numParts) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(numParts);
    *numParts = ptlist->nr_parts;
    return S_OK;
}

HRESULT MxCParticleTypeList_free(struct MxParticleTypeListHandle *handle) {
    MXPARTICLETYPELIST_GET(handle);
    ptlist->free();
    return S_OK;
}

HRESULT MxCParticleTypeList_insertI(struct MxParticleTypeListHandle *handle, int item, unsigned int *index) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(index);
    *index = ptlist->insert(item);
    return S_OK;
}

HRESULT MxCParticleTypeList_insertP(struct MxParticleTypeListHandle *handle, struct MxParticleTypeHandle *ptype, unsigned int *index) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(ptype); MXCPTRCHECK(ptype->MxObj);
    MXCPTRCHECK(index);
    *index = ptlist->insert((MxParticleType*)ptype->MxObj);
    return S_OK;
}

HRESULT MxCParticleTypeList_remove(struct MxParticleTypeListHandle *handle, int id) {
    MXPARTICLETYPELIST_GET(handle);
    return ptlist->remove(id);
}

HRESULT MxCParticleTypeList_extend(struct MxParticleTypeListHandle *handle, struct MxParticleTypeListHandle *other) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(other); MXCPTRCHECK(other->MxObj);
    ptlist->extend(*(MxParticleTypeList*)other->MxObj);
    return S_OK;
}

HRESULT MxCParticleTypeList_item(struct MxParticleTypeListHandle *handle, unsigned int i, struct MxParticleTypeHandle *item) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(item);
    MxParticleType *ptype = ptlist->item(i);
    MXCPTRCHECK(ptype);
    item->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCParticleTypeList_getAll(struct MxParticleTypeListHandle *handle) {
    MXCPTRCHECK(handle);
    handle->MxObj = (void*)MxParticleTypeList::all();
    return S_OK;
}

HRESULT MxCParticleTypeList_getVirial(struct MxParticleTypeListHandle *handle, float **virial) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(virial);
    auto _virial = ptlist->getVirial();
    MXMATRIX3_COPYFROM(_virial, (*virial));
    return S_OK;
}

HRESULT MxCParticleTypeList_getRadiusOfGyration(struct MxParticleTypeListHandle *handle, float *rog) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(rog);
    *rog = ptlist->getRadiusOfGyration();
    return S_OK;
}

HRESULT MxCParticleTypeList_getCenterOfMass(struct MxParticleTypeListHandle *handle, float **com) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(com);
    auto _com = ptlist->getCenterOfMass();
    MXVECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT MxCParticleTypeList_getCentroid(struct MxParticleTypeListHandle *handle, float **cent) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(cent);
    auto _cent = ptlist->getCentroid();
    MXVECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT MxCParticleTypeList_getMomentOfInertia(struct MxParticleTypeListHandle *handle, float **moi) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(moi);
    auto _moi = ptlist->getMomentOfInertia();
    MXMATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}

HRESULT MxCParticleTypeList_getPositions(struct MxParticleTypeListHandle *handle, float **positions) {
    MXPARTICLETYPELIST_GET(handle);
    return mx::capi::copyVecVecs3_2Arr(ptlist->getPositions(), positions);
}

HRESULT MxCParticleTypeList_getVelocities(struct MxParticleTypeListHandle *handle, float **velocities) {
    MXPARTICLETYPELIST_GET(handle);
    return mx::capi::copyVecVecs3_2Arr(ptlist->getVelocities(), velocities);
}

HRESULT MxCParticleTypeList_getForces(struct MxParticleTypeListHandle *handle, float **forces) {
    MXPARTICLETYPELIST_GET(handle);
    return mx::capi::copyVecVecs3_2Arr(ptlist->getForces(), forces);
}

HRESULT MxCParticleTypeList_sphericalPositions(struct MxParticleTypeListHandle *handle, float **coordinates) {
    MXPARTICLETYPELIST_GET(handle);
    return mx::capi::copyVecVecs3_2Arr(ptlist->sphericalPositions(), coordinates);
}

HRESULT MxCParticleTypeList_sphericalPositionsO(struct MxParticleTypeListHandle *handle, float *origin, float **coordinates) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(origin);
    MxVector3f _origin = MxVector3f::from(origin);
    return mx::capi::copyVecVecs3_2Arr(ptlist->sphericalPositions(&_origin), coordinates);
}

HRESULT MxCParticleTypeList_getParticles(struct MxParticleTypeListHandle *handle, struct MxParticleListHandle *plist) {
    MXPARTICLETYPELIST_GET(handle);
    MXCPTRCHECK(plist);
    MxParticleList *_plist = ptlist->particles();
    MXCPTRCHECK(_plist);
    plist->MxObj = (void*)_plist;
    return S_OK;
}

HRESULT MxCParticleTypeList_toString(struct MxParticleTypeListHandle *handle, char **str, unsigned int *numChars) {
    MXPARTICLETYPELIST_GET(handle);
    return mx::capi::str2Char(ptlist->toString(), str, numChars);
}

HRESULT MxCParticleTypeList_fromString(struct MxParticleTypeListHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxParticleTypeList *ptlist = MxParticleTypeList::fromString(str);
    handle->MxObj = (void*)ptlist;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCParticle_Verify() {
    return MxParticle_Verify();
}

HRESULT MxCParticleType_FindFromName(struct MxParticleTypeHandle *handle, const char* name) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(name);
    MxParticleType *ptype = MxParticleType_FindFromName(name);
    MXCPTRCHECK(ptype);
    handle->MxObj = (void*)ptype;
    return S_OK;
}

/**
 * @brief Get a registered particle type by type id
 * 
 * @param handle handle to populate
 * @param name name of cluster type
 * @return S_OK on success 
 */
HRESULT MxCParticleType_getFromId(struct MxParticleTypeHandle *handle, unsigned int pid) {
    MXCPTRCHECK(handle);
    if(pid >= _Engine.nr_types) 
        return E_FAIL;
    
    MxParticleType *ptype = &_Engine.types[pid];
    MXCPTRCHECK(ptype);
    
    handle->MxObj = (void*)ptype;
    return S_OK;
}

unsigned int *MxCParticle_Colors() {
    return MxParticle_Colors;
}
