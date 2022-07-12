/**
 * @file MxMeshSurface.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh surface class
 * @date 2022-26-04
 * 
 */

#include "MxMeshSurface.h"

#include "MxMeshVertex.h"
#include "MxMeshSurface.h"
#include "MxMeshBody.h"
#include "MxMeshStructure.h"
#include "MxMeshSolver.h"

#include <Magnum/Math/Math.h>

#include <MxLogger.h>
#include <MxUtil.h>
#include <metrics.h>

#include <io/Mx3DFVertexData.h>
#include <io/Mx3DFEdgeData.h>


#define MXMESHSURFACE_VERTEXINDEX(vertices, idx) idx >= vertices.size() ? idx - vertices.size() : (idx < 0 ? idx + vertices.size() : idx)


MxVector3f triNorm(const MxVector3f &p1, const MxVector3f &p2, const MxVector3f &p3) {
    return Magnum::Math::cross(p1 - p2, p3 - p2);
}

MxMeshSurface::MxMeshSurface() : 
    MxMeshObj(), 
    b1{NULL}, 
    b2{NULL}, 
    area{0.f}, 
    _volumeContr{0.f}, 
    style{NULL}
{}

MxMeshSurface::MxMeshSurface(std::vector<MxMeshVertex*> _vertices) : 
    MxMeshSurface()
{
    if(_vertices.size() >= 3) {
        for(auto &v : _vertices) {
            addParent(v);
            v->addChild(this);
        }
    } 
    else {
        Log(LOG_ERROR) << "Surfaces require at least 3 vertices (" << _vertices.size() << " given)";
    }
}

static HRESULT MxMeshSurface_order3DFFaceVertices(Mx3DFFaceData *face, std::vector<Mx3DFVertexData*> &result) {
    auto vedges = face->getEdges();
    auto vverts = face->getVertices();
    result.clear();
    std::vector<int> edgesLeft;
    for(int i = 1; i < vedges.size(); edgesLeft.push_back(i), i++) {}
    
    Mx3DFVertexData *currentVertex;
    Mx3DFEdgeData *edge = vedges[0];
    currentVertex = edge->vb;
    result.push_back(edge->va);
    result.push_back(currentVertex);

    while(edgesLeft.size() > 0) { 
        int j;
        edge = 0;
        for(j = 0; j < edgesLeft.size(); j++) {
            edge = vedges[edgesLeft[j]];
            if(edge->va == currentVertex || edge->vb == currentVertex) 
                break;
        }
        if(!edge) {
            Log(LOG_ERROR) << "Error importing face";
            return E_FAIL;
        } 
        else {
            currentVertex = edge->va == currentVertex ? edge->vb : edge->va;
            result.push_back(currentVertex);
            edgesLeft.erase(std::find(edgesLeft.begin(), edgesLeft.end(), edgesLeft[j]));
        }
    }
    return S_OK;
}

MxMeshSurface::MxMeshSurface(Mx3DFFaceData *face) : 
    MxMeshSurface()
{
    std::vector<Mx3DFVertexData*> vverts;
    if(MxMeshSurface_order3DFFaceVertices(face, vverts) == S_OK) {
        for(auto &vv : vverts) {
            MxMeshVertex *v = new MxMeshVertex(vv);
            addParent(v);
            v->addChild(this);
        }
    }
}

std::vector<MxMeshObj*> MxMeshSurface::children() {
    std::vector<MxMeshObj*> result;
    if(b1) 
        result.push_back((MxMeshObj*)b1);
    if(b2) 
        result.push_back((MxMeshObj*)b2);
    return result;
}

HRESULT MxMeshSurface::addChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::BODY)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshBody *b = (MxMeshBody*)obj;
    
    if(b1) {
        if(b1 == b) {
            Log(LOG_ERROR);
            return E_FAIL;
        }

        if(b2) {
            if(b2 == b) {
                Log(LOG_ERROR);
                return E_FAIL;
            }
        }
        else {
            b2 = b;
        }
    }
    else 
        b1 = b;

    return S_OK;
}

HRESULT MxMeshSurface::addParent(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::VERTEX)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshVertex *v = (MxMeshVertex*)obj;
    if(std::find(vertices.begin(), vertices.end(), v) != vertices.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.push_back(v);
    return S_OK;
}

HRESULT MxMeshSurface::removeChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::BODY)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshBody *b = (MxMeshBody*)obj;

    if(b1 == b) 
        b1 = NULL;
    else if(b2 == b) 
        b2 = NULL;
    else {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT MxMeshSurface::removeParent(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::VERTEX)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshVertex *v = (MxMeshVertex*)obj;
    auto itr = std::find(vertices.begin(), vertices.end(), v);
    if(itr == vertices.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    vertices.erase(itr);
    return S_OK;
}

bool MxMeshSurface::validate() {
    return vertices.size() >= 3;
}

HRESULT MxMeshSurface::refreshBodies() {
    if(!b1 && !b2) 
        return S_OK;
    else if(normal.isZero()) {
        Log(LOG_ERROR) << "Normal not set";
        return E_FAIL;
    } 
    else if(centroid.isZero()) {
        Log(LOG_ERROR) << "Centroid not set";
        return E_FAIL;
    }

    MxMeshBody *bo = NULL;
    MxMeshBody *bi = NULL;

    MxVector3f n;
    if(b1) {
        n = centroid - b1->getCentroid();
        if(n.dot(normal) > 0) 
            bo = b1;
        else 
            bi = b1;
    }
    if(b2) {
        n = centroid - b2->getCentroid();
        if(n.dot(normal) > 0) 
            bo = b2;
        else 
            bi = b2;
    }

    b1 = bo;
    b2 = bi;

    return S_OK;
}

MxMeshSurfaceType *MxMeshSurface::type() {
    MxMeshSolver *solver = MxMeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getSurfaceType(typeId);
}

std::vector<MxMeshStructure*> MxMeshSurface::getStructures() {
    std::vector<MxMeshStructure*> result;
    if(b1) 
        for(auto &s : b1->getStructures()) 
            result.push_back(s);
    if(b2) 
        for(auto &s : b2->getStructures()) 
            result.push_back(s);
    return unique(result);
}

std::vector<MxMeshBody*> MxMeshSurface::getBodies() {
    std::vector<MxMeshBody*> result;
    if(b1) 
        result.push_back(b1);
    if(b2) 
        result.push_back(b2);
    return result;
}

MxMeshVertex *MxMeshSurface::findVertex(const MxVector3f &dir) {
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

MxMeshBody *MxMeshSurface::findBody(const MxVector3f &dir) {
    MxMeshBody *result = 0;

    MxVector3f pta = centroid;
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

std::vector<MxMeshVertex*> MxMeshSurface::neighborVertices(MxMeshVertex *v) {
    std::vector<MxMeshVertex*> result(2, 0);

    MxMeshVertex *vo;
    for(auto itr = vertices.begin(); itr != vertices.end(); itr++) {
        if(*itr == v) {
            result[0] = itr + 1 == vertices.end() ? *vertices.begin() : *(itr + 1);
            result[1] = itr == vertices.begin() ? *(vertices.end() - 1) : *(itr - 1);
            break;
        }
    }

    return result;
}

std::vector<MxMeshSurface*> MxMeshSurface::neighborSurfaces() { 
    std::vector<MxMeshSurface*> result;
    if(b1) 
        for(auto &s : b1->neighborSurfaces(this)) 
            result.push_back(s);
    if(b2) 
        for(auto &s : b2->neighborSurfaces(this)) 
            result.push_back(s);
    return unique(result);
}

std::vector<unsigned int> MxMeshSurface::contiguousEdgeLabels(MxMeshSurface *other) {
    std::vector<MxMeshVertex*> overtices = mx::models::vertex::vectorToDerived<MxMeshVertex>(other->parents());
    std::vector<bool> sharedVertices(vertices.size(), false);
    for(unsigned int i = 0; i < sharedVertices.size(); i++) 
        if(std::find(overtices.begin(), overtices.end(), vertices[i]) != overtices.end()) 
            sharedVertices[i] = true;

    bool shared_c, shared_n;
    std::vector<unsigned int> result(vertices.size(), 0);
    unsigned int edgeLabel = 1;
    for(unsigned int i = 0; i < sharedVertices.size(); i++) {
        shared_c = sharedVertices[i];
        shared_n = sharedVertices[i + 1 == sharedVertices.size() ? 0 : i + 1];
        if(shared_c) 
            result[i] = edgeLabel;
        if(shared_c && !shared_n) 
            edgeLabel++;
    }

    if(result[0] > 0 && result[result.size() - 1] > 0) {
        unsigned int lastLabel = result[result.size() - 1];
        for(unsigned int i = 0; i < result.size(); i++) 
            if(result[i] == lastLabel) 
                result[i] = result[0];
    }

    return result;
}

unsigned int MxMeshSurface::numSharedContiguousEdges(MxMeshSurface *other) {
    unsigned int result = 0;
    for(auto &i : contiguousEdgeLabels(other)) 
        result = std::max(result, i);
    return result;
}

float MxMeshSurface::volumeSense(MxMeshBody *body) {
    if(body == b1) 
        return 1.f;
    else if(body == b2) 
        return -1.f;
    return 0.f;
}

float MxMeshSurface::getVertexArea(MxMeshVertex *v) {
    float result = 0.f;
    
    for(unsigned int i = 0; i < vertices.size(); i++) {
        MxMeshVertex *vc = vertices[i];
        MxMeshVertex *vn = vertices[MXMESHSURFACE_VERTEXINDEX(vertices, i + 1)];

        if(vc == v || vn == v) {
            MxVector3f triNormal = triNorm(vc->getPosition(), centroid, vn->getPosition());
            result += triNormal.length();
        }
    }

    return result / 4.f;
}

MxVector3f MxMeshSurface::triangleNormal(const unsigned int &idx) {
    return triNorm(vertices[idx]->getPosition(), 
                   centroid, 
                   vertices[MXMESHSURFACE_VERTEXINDEX(vertices, idx + 1)]->getPosition());
}

HRESULT MxMeshSurface::positionChanged() {
    normal = MxVector3f(0.f);
    centroid = MxVector3f(0.f);
    area = 0.f;
    _volumeContr = 0.f;

    for(auto &v : vertices) 
        centroid += v->getPosition();
    centroid /= (float)vertices.size();

    for(unsigned int i = 0; i < vertices.size(); i++) {
        MxVector3f triNormal = triangleNormal(i);

        _volumeContr += triNormal.dot(centroid);
        area += triNormal.length();
        normal += triNormal;
    }

    normal = normal.normalized();
    area /= 2.f;
    _volumeContr /= 6.f;

    return S_OK;
}

MxMeshSurface *MxMeshSurfaceType::operator() (std::vector<MxMeshVertex*> _vertices) {
    MxMeshSurface *s = new MxMeshSurface(_vertices);
    s->typeId = this->id;
    return s;
}

MxMeshSurface *MxMeshSurfaceType::operator() (const std::vector<MxVector3f> &_positions) {
    std::vector<MxMeshVertex*> _vertices;
    for(auto &p : _positions) 
        _vertices.push_back(new MxMeshVertex(p));
    
    Log(LOG_DEBUG) << "Created " << _vertices.size() << " vertices";
    
    return (*this)(_vertices);
}

MxMeshSurface *MxMeshSurfaceType::operator() (Mx3DFFaceData *face) {
    std::vector<Mx3DFVertexData*> vverts;
    std::vector<MxVector3f> _positions;
    if(MxMeshSurface_order3DFFaceVertices(face, vverts) == S_OK) 
        for(auto &vv : vverts) 
            _positions.push_back(vv->position);
    
    return (*this)(_positions);
}
