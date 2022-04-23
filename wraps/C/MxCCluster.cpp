/**
 * @file MxCCluster.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxCluster
 * @date 2022-03-29
 */

#include "MxCCluster.h"

#include "mechanica_c_private.h"
#include "MxCParticle.h"

#include <MxCluster.hpp>
#include <engine.h>


namespace mx { 

MxClusterParticleHandle *castC(struct MxClusterParticleHandleHandle *handle) {
    return castC<MxClusterParticleHandle, MxClusterParticleHandleHandle>(handle);
}

MxClusterParticleType *castC(struct MxClusterParticleTypeHandle *handle) {
    return castC<MxClusterParticleType, MxClusterParticleTypeHandle>(handle);
}

}

#define MXCLUSTERHANDLE_GET(handle) \
    MxClusterParticleHandle *phandle = mx::castC<MxClusterParticleHandle, MxClusterParticleHandleHandle>(handle); \
    MXCPTRCHECK(phandle);

#define MXCTYPEHANDLE_GET(handle) \
    MxClusterParticleType *ptype = mx::castC<MxClusterParticleType, MxClusterParticleTypeHandle>(handle); \
    MXCPTRCHECK(ptype);


////////////////////
// MxCClusterType //
////////////////////


struct MxCClusterType MxCClusterTypeDef_init() {
    struct MxCClusterType MxCClusterTypeDef = {
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
        0, 
        NULL
    };
    return MxCClusterTypeDef;
}


/////////////////////////////
// MxClusterParticleHandle //
/////////////////////////////


HRESULT MxCClusterParticleHandle_init(struct MxClusterParticleHandleHandle *handle, int id) {
    MXCPTRCHECK(handle);
    if(id >= _Engine.s.size_parts) 
        return E_FAIL;

    MxParticle *p = _Engine.s.partlist[id];
    MXCPTRCHECK(p);

    MxClusterParticleHandle *phandle = new MxClusterParticleHandle(id, p->typeId);
    handle->MxObj = (void*)phandle;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_destroy(struct MxClusterParticleHandleHandle *handle) {
    return mx::capi::destroyHandle<MxClusterParticleHandle, MxClusterParticleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCClusterParticleHandle_createParticle(struct MxClusterParticleHandleHandle *handle, 
                                                struct MxParticleTypeHandle *partTypeHandle, 
                                                int *pid, 
                                                float **position, 
                                                float **velocity) {
    MXCLUSTERHANDLE_GET(handle); 
    MXCPTRCHECK(partTypeHandle); MXCPTRCHECK(partTypeHandle->MxObj);
    MXCPTRCHECK(pid);
    MxParticleType *partType = (MxParticleType*)partTypeHandle->MxObj;

    MxVector3f pos, vel, *pos_p = NULL, *vel_p = NULL;
    if(position) {
        pos = MxVector3f::from(*position);
        pos_p = &pos;
    }
    if(velocity) {
        vel = MxVector3f::from(*velocity);
        vel_p = &vel;
    }

    auto p = (*phandle)(partType, pos_p, vel_p);
    MXCPTRCHECK(p);
    
    *pid = p->id;

    return S_OK;
}

HRESULT MxCClusterParticleHandle_createParticleS(struct MxClusterParticleHandleHandle *handle, 
                                                 struct MxParticleTypeHandle *partTypeHandle, 
                                                 int *pid, 
                                                 const char *str) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(partTypeHandle); MXCPTRCHECK(!partTypeHandle->MxObj);
    MXCPTRCHECK(pid);
    MXCPTRCHECK(str);
    MxParticleType *partType = (MxParticleType*)partTypeHandle->MxObj;
    auto p = (*phandle)(partType, std::string(str));
    MXCPTRCHECK(p);
    *pid = p->id;
    delete p;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_splitAxis(struct MxClusterParticleHandleHandle *handle, int *cid, float *axis, float time) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(cid);
    MXCPTRCHECK(axis);
    MxVector3f _axis = MxVector3f::from(axis);
    auto c = phandle->split(&_axis, 0, 0);
    MXCPTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_splitRand(struct MxClusterParticleHandleHandle *handle, int *cid, float time) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(cid);
    auto c = phandle->split();
    MXCPTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_split(struct MxClusterParticleHandleHandle *handle, int *cid, float time, float *normal, float *point) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(cid);
    MXCPTRCHECK(normal);
    MXCPTRCHECK(point);
    MxVector3f _normal = MxVector3f::from(normal);
    MxVector3f _point = MxVector3f::from(point);
    auto c = phandle->split(0, 0, 0, &_normal, &_point);
    MXCPTRCHECK(c);
    *cid = c->id;
    delete c;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getNumParts(struct MxClusterParticleHandleHandle *handle, int *numParts) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(numParts);
    MxParticle *p = phandle->part();
    MXCPTRCHECK(p);
    *numParts = p->nr_parts;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getParticle(struct MxClusterParticleHandleHandle *handle, int i, struct MxParticleHandleHandle *parthandle) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(parthandle);
    MxParticle *cp = phandle->part();
    MXCPTRCHECK(cp);
    if(i >= cp->nr_parts) 
        return E_FAIL;
    int pid = cp->parts[i];
    if(pid >= _Engine.s.size_parts) 
        return E_FAIL;
    MxParticle *p = _Engine.s.partlist[pid];
    MXCPTRCHECK(p);
    MxParticleHandle *ph = new MxParticleHandle(p->id, p->typeId);
    parthandle->MxObj = (void*)ph;
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getRadiusOfGyration(struct MxClusterParticleHandleHandle *handle, float *radiusOfGyration) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(radiusOfGyration);
    *radiusOfGyration = phandle->getRadiusOfGyration();
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getCenterOfMass(struct MxClusterParticleHandleHandle *handle, float **com) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(com);
    MxVector3f _com = phandle->getCenterOfMass();
    MXVECTOR3_COPYFROM(_com, (*com));
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getCentroid(struct MxClusterParticleHandleHandle *handle, float **cent) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(cent);
    MxVector3f _cent = phandle->getCentroid();
    MXVECTOR3_COPYFROM(_cent, (*cent));
    return S_OK;
}

HRESULT MxCClusterParticleHandle_getMomentOfInertia(struct MxClusterParticleHandleHandle *handle, float **moi) {
    MXCLUSTERHANDLE_GET(handle);
    MXCPTRCHECK(moi);
    MxMatrix3f _moi = phandle->getMomentOfInertia();
    MXMATRIX3_COPYFROM(_moi, (*moi));
    return S_OK;
}


///////////////////////////
// MxClusterParticleType //
///////////////////////////


HRESULT MxCClusterParticleType_init(struct MxClusterParticleTypeHandle *handle) {
    MXCPTRCHECK(handle);
    MxClusterParticleType *ptype = new MxClusterParticleType(true);
    handle->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCClusterParticleType_initD(struct MxClusterParticleTypeHandle *handle, struct MxCClusterType pdef) {
    HRESULT result = MxCClusterParticleType_init(handle);
    if(result != S_OK) 
        return result;
    MXCTYPEHANDLE_GET(handle);

    ptype->mass = pdef.mass;
    ptype->radius = pdef.radius;
    if(pdef.target_energy) ptype->target_energy = *pdef.target_energy;
    ptype->minimum_radius = pdef.minimum_radius;
    ptype->dynamics = pdef.dynamics;
    ptype->setFrozen(pdef.frozen);
    if(pdef.name) {
        std::string ns(pdef.name);
        std::strncpy(ptype->name, ns.c_str(), ns.size());
    }
    if(pdef.numTypes > 0) {
        MxParticleTypeHandle *pth;
        for(unsigned int i = 0; i < pdef.numTypes; i++) {
            pth = pdef.types[i];
            MXCPTRCHECK(pth); MXCPTRCHECK(pth->MxObj);
            ptype->types.insert((MxParticleType*)pth->MxObj);
        }
    }

    return S_OK;
}

HRESULT MxCClusterParticleType_addType(struct MxClusterParticleTypeHandle *handle, struct MxParticleTypeHandle *phandle) {
    MXCTYPEHANDLE_GET(handle);
    MXCPTRCHECK(phandle); MXCPTRCHECK(phandle->MxObj);
    MxParticleType *partType = (MxParticleType*)phandle->MxObj;
    ptype->types.insert(partType->id);
    return S_OK;
}

HRESULT MxCClusterParticleType_hasType(struct MxClusterParticleTypeHandle *handle, struct MxParticleTypeHandle *phandle, bool *hasType) {
    MXCTYPEHANDLE_GET(handle);
    MXCPTRCHECK(phandle); MXCPTRCHECK(phandle->MxObj);
    MXCPTRCHECK(hasType);
    MxParticleType *partType = (MxParticleType*)phandle->MxObj;
    *hasType = ptype->hasType(partType);
    return S_OK;
}

HRESULT MxCClusterParticleType_registerType(struct MxClusterParticleTypeHandle *handle) {
    MXCTYPEHANDLE_GET(handle);
    return ptype->registerType();
}

HRESULT MxCClusterParticleType_createParticle(struct MxClusterParticleTypeHandle *handle, int *pid, float *position, float *velocity) {
    MXCTYPEHANDLE_GET(handle);
    MxParticleTypeHandle _handle;
    _handle.MxObj = (void*)ptype;
    return MxCParticleType_createParticle(&_handle, pid, position, velocity);
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCClusterParticleType_FindFromName(struct MxClusterParticleTypeHandle *handle, const char* name) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(name);
    MxClusterParticleType *ptype = MxClusterParticleType_FindFromName(name);
    MXCPTRCHECK(ptype);
    handle->MxObj = (void*)ptype;
    return S_OK;
}

HRESULT MxCClusterParticleType_getFromId(struct MxClusterParticleTypeHandle *handle, unsigned int pid) {
    MXCPTRCHECK(handle);
    if(pid >= _Engine.nr_types) 
        return E_FAIL;
    
    MxParticleType *ptype = &_Engine.types[pid];
    MXCPTRCHECK(ptype);
    if(!ptype->isCluster()) 
        return E_FAIL;
    
    handle->MxObj = (void*)ptype;
    return S_OK;
}
