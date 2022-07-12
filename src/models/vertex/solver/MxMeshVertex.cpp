/**
 * @file MxMeshVertex.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh vertex class
 * @date 2022-26-04
 * 
 */

#include "MxMeshVertex.h"

#include "MxMeshSurface.h"
#include "MxMeshBody.h"
#include "MxMeshStructure.h"

#include <MxLogger.h>
#include <engine.h>
#include <MxUtil.h>


MxMeshParticleType *MxMeshParticleType_get() {
    Log(LOG_TRACE);

    MxMeshParticleType tmp;
    MxParticleType *result = MxParticleType_FindFromName(tmp.name);
    if(result) 
        return (MxMeshParticleType*)result;
    
    Log(LOG_DEBUG) << "Registering vertex particle type with name " << tmp.name;
    tmp.registerType();
    Log(LOG_DEBUG) << "Particle types: " << _Engine.nr_types;
    
    result = MxParticleType_FindFromName(tmp.name);
    if(!result) {
        Log(LOG_ERROR);
        return NULL;
    }
    return (MxMeshParticleType*)result;
}

std::vector<MxMeshVertex*> MxMeshVertex::neighborVertices() {
    std::vector<MxMeshVertex*> result;

    for(auto &s : surfaces) 
        for(auto &v : s->neighborVertices(this)) 
            result.push_back(v);
    return unique(result);
}

std::vector<MxMeshSurface*> MxMeshVertex::sharedSurfaces(MxMeshVertex *other) {
    std::vector<MxMeshSurface*> result;
    for(auto &s : surfaces) 
        if(other->in(s) && std::find(result.begin(), result.end(), s) == result.end()) 
            result.push_back(s);
    return result;
}

float MxMeshVertex::getVolume() {
    float result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexVolume(this);
    return result;
}

float MxMeshVertex::getMass() {
    float result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexMass(this);
    return result;
}

HRESULT MxMeshVertex::positionChanged() {
    return S_OK;
}

HRESULT MxMeshVertex::updateProperties() {
    MxParticleHandle *p = particle();
    const float vMass = getMass();
    if(p && vMass > 0.f) {
        p->setMass(vMass);
    }
    return S_OK;
}

MxParticleHandle *MxMeshVertex::particle() {
    if(this->pid < 0) {
        Log(LOG_DEBUG);
        return NULL;
    }

    MxParticle *p = MxParticle_FromId(this->pid);
    if(!p) {
        Log(LOG_ERROR);
        return NULL;
    }

    return p->py_particle();
}

MxVector3f MxMeshVertex::getPosition() {
    auto p = particle();
    if(!p) { 
        Log(LOG_ERROR) << "No assigned particle.";
        MxVector3f(-1.f, -1.f, -1.f);
    }
    return p->getPosition();
}

HRESULT MxMeshVertex::setPosition(const MxVector3f &pos) {
    auto p = particle();
    if(!p) {
        Log(LOG_ERROR) << "No assigned particle.";
        return E_FAIL;
    }
    p->setPosition(pos);

    for(auto &s : surfaces) 
        s->positionChanged();

    return S_OK;
}

MxMeshVertex::MxMeshVertex() : 
    MxMeshObj(), 
    pid{-1}
{}

MxMeshVertex::MxMeshVertex(const unsigned int &_pid) : 
    MxMeshVertex() 
{
    pid = (int)_pid;
};

MxMeshVertex::MxMeshVertex(const MxVector3f &position) : 
    MxMeshVertex()
{
    MxMeshParticleType *ptype = MxMeshParticleType_get();
    if(!ptype) {
        Log(LOG_ERROR) << "Could not instantiate particle type";
        this->pid = -1;
    } 
    else {
        MxVector3f _position = position;
        MxParticleHandle *ph = (*ptype)(&_position);
        this->pid = ph->id;
    }
}

MxMeshVertex::MxMeshVertex(Mx3DFVertexData *vdata) :
    MxMeshVertex(vdata->position)
{}

std::vector<MxMeshObj*> MxMeshVertex::children() {
    return mx::models::vertex::vectorToBase(surfaces);
}

HRESULT MxMeshVertex::addChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::SURFACE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshSurface *s = (MxMeshSurface*)obj;
    if(std::find(surfaces.begin(), surfaces.end(), s) != surfaces.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    surfaces.push_back(s);
    return S_OK;
}

HRESULT MxMeshVertex::removeChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::SURFACE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshSurface *s = (MxMeshSurface*)obj;
    auto itr = std::find(surfaces.begin(), surfaces.end(), s);
    if(itr == surfaces.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }
    surfaces.erase(itr);
    return S_OK;
}

std::vector<MxMeshStructure*> MxMeshVertex::getStructures() {
    std::vector<MxMeshStructure*> result;
    for(auto &s : surfaces) 
        for(auto &ss : s->getStructures()) 
            result.push_back(ss);
    return unique(result);
}

std::vector<MxMeshBody*> MxMeshVertex::getBodies() {
    std::vector<MxMeshBody*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            result.push_back(b);
    return unique(result);
}

MxMeshSurface *MxMeshVertex::findSurface(const MxVector3f &dir) {
    MxMeshSurface *result = 0;

    MxVector3f pta = getPosition();
    MxVector3f ptb = pta + dir;
    float bestDist2 = 0;

    for(auto &s : getSurfaces()) {
        MxVector3f pt = s->getCentroid();
        float dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = s;
            bestDist2 = dist2;
        }
    }

    return result;
}

MxMeshBody *MxMeshVertex::findBody(const MxVector3f &dir) {
    MxMeshBody *result = 0;

    MxVector3f pta = getPosition();
    MxVector3f ptb = pta + dir;
    float bestDist2 = 0;

    for(auto &b : getBodies()) {
        MxVector3f pt = b->getCentroid();
        float dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = b;
            bestDist2 = dist2;
        }
    }

    return result;
}
