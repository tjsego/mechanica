/**
 * @file MxMeshBody.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh body class
 * @date 2022-26-04
 * 
 */

#include "MxMeshBody.h"

#include "MxMeshVertex.h"
#include "MxMeshSurface.h"
#include "MxMeshStructure.h"
#include "MxMeshSolver.h"

#include <MxLogger.h>
#include <MxUtil.h>

#include <Magnum/Math/Math.h>


void MxMeshBody::_updateInternal() {
    for(auto &v : getVertices()) 
        v->positionChanged();
    for(auto &s : getSurfaces()) 
        s->positionChanged();

    centroid = MxVector3f(0.f);
    area = 0.f;
    volume = 0.f;

    for(auto &s : surfaces) {
        centroid += s->getCentroid() * s->getArea();
        area += s->getArea();
    }
    centroid /= area;

    for(auto &s : surfaces) {
        s->refreshBodies();
        volume += s->getVolumeContr(this);
    }

}


MxMeshBody::MxMeshBody() : 
    MxMeshObj(), 
    centroid{0.f}, 
    area{0.f}, 
    volume{0.f}, 
    density{0.f}
{}

MxMeshBody::MxMeshBody(std::vector<MxMeshSurface*> _surfaces) 
    : MxMeshBody() 
{
    if(_surfaces.size() >= 4) {
        for(auto &s : _surfaces) {
            addParent(s);
            s->addChild(this);
        }
        _updateInternal();
    } 
    else {
        Log(LOG_ERROR) << "A body requires at least 4 surfaces";
    }
};

std::vector<MxMeshObj*> MxMeshBody::parents() { return mx::models::vertex::vectorToBase(surfaces); }

std::vector<MxMeshObj*> MxMeshBody::children() { return mx::models::vertex::vectorToBase(structures); }

HRESULT MxMeshBody::addChild(MxMeshObj *obj) { 
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshStructure *s = (MxMeshStructure*)obj;
    if(std::find(structures.begin(), structures.end(), s) != structures.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    structures.push_back(s);
    return S_OK;
}

HRESULT MxMeshBody::addParent(MxMeshObj *obj) {
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

HRESULT MxMeshBody::removeChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshStructure *s = (MxMeshStructure*)obj;
    auto itr = std::find(structures.begin(), structures.end(), s);
    if(itr == structures.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    structures.erase(itr);
    return S_OK;
}

HRESULT MxMeshBody::removeParent(MxMeshObj *obj) {
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

bool MxMeshBody::validate() {
    return surfaces.size() >= 3;
}

HRESULT MxMeshBody::positionChanged() { 
    centroid = MxVector3f(0.f);
    area = 0.f;
    volume = 0.f;

    for(auto &s : surfaces) {
        centroid += s->getCentroid() * s->getArea();
        area += s->getArea();
        volume += s->getVolumeContr(this);
    }
    centroid /= area;

    return S_OK;
}

MxMeshBodyType *MxMeshBody::type() {
    MxMeshSolver *solver = MxMeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getBodyType(typeId);
}

std::vector<MxMeshStructure*> MxMeshBody::getStructures() {
    std::vector<MxMeshStructure*> result;
    for(auto &s : structures) {
        result.push_back(s);
        for(auto &ss : s->getStructures()) 
            result.push_back(ss);
    }
    return unique(result);
}

std::vector<MxMeshVertex*> MxMeshBody::getVertices() {
    std::vector<MxMeshVertex*> result;

    for(auto &s : surfaces) 
        for(auto &v : s->vertices) 
            result.push_back(v);

    return unique(result);
}

MxMeshVertex *MxMeshBody::findVertex(const MxVector3f &dir) {
    MxMeshVertex *result = 0;

    MxVector3f pta = centroid;
    MxVector3f ptb = pta + dir;
    float bestDist2 = 0;

    for(auto &v : getVertices()) {
        MxVector3f pt = v->getPosition();
        float dist2 = Magnum::Math::Distance::linePointSquared(pta, ptb, pt);
        if((!result || dist2 <= bestDist2) && dir.dot(pt - pta) >= 0.f) { 
            result = v;
            bestDist2 = dist2;
        }
    }

    return result;
}

MxMeshSurface *MxMeshBody::findSurface(const MxVector3f &dir) {
    MxMeshSurface *result = 0;

    MxVector3f pta = centroid;
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

std::vector<MxMeshBody*> MxMeshBody::neighborBodies() {
    std::vector<MxMeshBody*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            if(b != this) 
                result.push_back(b);
    return unique(result);
}

std::vector<MxMeshSurface*> MxMeshBody::neighborSurfaces(MxMeshSurface *s) {
    std::vector<MxMeshSurface*> result;
    for(auto &so : surfaces) {
        if(so == s) 
            continue;
        for(auto &v : s->vertices) 
            if(v->in(so)) { 
                result.push_back(so);
                break;
            }
    }
    return unique(result);
}

MxVector3f MxMeshBody::getVelocity() {
    MxVector3f result;
    for(auto &v : getVertices()) 
        result += v->particle()->getVelocity() * getVertexMass(v);
    return result / getMass();
}

float MxMeshBody::getVertexArea(MxMeshVertex *v) {
    float result;
    for(auto &s : surfaces) 
        result += s->getVertexArea(v);
    return result;
}

float MxMeshBody::getVertexVolume(MxMeshVertex *v) {
    if(area == 0.f) 
        return 0.f;
    return getVertexArea(v) / area * volume;
}

float MxMeshBody::contactArea(MxMeshBody *other) {
    float result = 0.f;
    for(auto &s : surfaces) 
        if(std::find(other->surfaces.begin(), other->surfaces.end(), s) != other->surfaces.end()) 
            result += s->area;
    return result;
}

MxMeshBody *MxMeshBodyType::operator() (std::vector<MxMeshSurface*> surfaces) {
    // Verify that at least 4 surfaces are given
    if(surfaces.size() < 4) {
        Log(LOG_ERROR) << "A body requires at least 4 surfaces";
        return NULL;
    }
    // Verify that every parent vertex is in at least two given surfaces
    // todo: current vertex condition is necessary for body construction, but is it sufficient?
    for(unsigned int i = 0; i < surfaces.size(); i++) 
        for(auto &pv : surfaces[i]->parents()) {
            bool twiceConnected = false;
            for(unsigned int j = 0; j < surfaces.size(); j++) {
                if(i == j) 
                    continue;

                if(pv->in(surfaces[j])) {
                    twiceConnected = true;
                    break;
                }
            }
            if(!twiceConnected) {
                Log(LOG_ERROR) << "Detected insufficient connectivity";
                return NULL;
            }
        }

    MxMeshBody *b = new MxMeshBody(surfaces);
    b->typeId = this->id;
    b->density = this->density;
    return b;
}
