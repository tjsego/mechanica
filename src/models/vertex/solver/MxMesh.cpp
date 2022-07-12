/**
 * @file MxMesh.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh class
 * @date 2022-26-04
 * 
 */

#include "MxMesh.h"

#include "MxMeshSolver.h"

#include <engine.h>
#include <MxLogger.h>

#include <algorithm>


#define MXMESH_GETPART(idx, inv) idx >= inv.size() ? NULL : inv[idx]


HRESULT MxMesh_checkUnstoredObj(MxMeshObj *obj) {
    if(!obj || obj->objId >= 0 || obj->mesh) {
        Log(LOG_ERROR);
        return E_FAIL;
    }
    
    return S_OK;
}


HRESULT MxMesh_checkStoredObj(MxMeshObj *obj, MxMesh *mesh) {
    if(!obj || obj->objId < 0 || obj->mesh == NULL || obj->mesh != mesh) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


template <typename T> 
HRESULT MxMesh_checkObjStorage(MxMeshObj *obj, std::vector<T*> inv) {
    if(!MxMesh_checkStoredObj(obj) || obj->objId >= inv.size()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    return S_OK;
}


template <typename T> 
int MxMesh_invIdAndAlloc(std::vector<T*> &inv, std::set<unsigned int> &availIds, T *obj=NULL) {
    if(!availIds.empty()) {
        std::set<unsigned int>::iterator itr = availIds.begin();
        unsigned int objId = *itr;
        if(obj) {
            inv[objId] = obj;
            obj->objId = *itr;
        }
        availIds.erase(itr);
        return objId;
    }
    
    int res = inv.size();
    int new_size = inv.size() + MXMESHINV_INCR;
    inv.reserve(new_size);
    for(unsigned int i = res; i < new_size; i++) {
        inv.push_back(NULL);
        if(i != res) 
            availIds.insert(i);
    }
    if(obj) {
        inv[res] = obj;
        obj->objId = res;
    }
    return res;
}


#define MXMESH_OBJINVCHECK(obj, inv) \
    { \
        if(obj->objId >= inv.size()) { \
            Log(LOG_ERROR) << "Object with id " << obj->objId << " exceeds inventory (" << inv.size() << ")"; \
            return E_FAIL; \
        } \
    }


template <typename T> 
HRESULT MxMesh_addObj(MxMesh *mesh, std::vector<T*> &inv, std::set<unsigned int> &availIds, T *obj) {
    if(MxMesh_checkUnstoredObj(obj) != S_OK || !obj->validate()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    obj->objId = MxMesh_invIdAndAlloc(inv, availIds, obj);
    obj->mesh = mesh;
    return S_OK;
}

HRESULT MxMesh::add(MxMeshVertex *obj) { 
    isDirty = true;
    if(_solver) 
        _solver->setDirty(true);

    return MxMesh_addObj(this, this->vertices, this->vertexIdsAvail, obj);
}

HRESULT MxMesh::add(MxMeshSurface *obj){ 
    isDirty = true;

    if(MxMesh_addObj(this, this->surfaces, this->surfaceIdsAvail, obj) != S_OK) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    for(auto &p : obj->parents()) 
        if(p->objId < 0 && add((MxMeshVertex*)p) != S_OK) {
            Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

HRESULT MxMesh::add(MxMeshBody *obj){ 
    isDirty = true;

    if(MxMesh_addObj(this, this->bodies, this->bodyIdsAvail, obj) != S_OK) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    for(auto &p : obj->parents()) 
        if(p->objId < 0 && add((MxMeshSurface*)p) != S_OK) {
            Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

HRESULT MxMesh::add(MxMeshStructure *obj){ 
    isDirty = true;

    if(MxMesh_addObj(this, this->structures, this->structureIdsAvail, obj) != S_OK) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    for(auto &p : obj->parents()) {
        if(p->objId >= 0) 
            continue;

        if(mx::models::vertex::check(p, MxMeshObj::Type::STRUCTURE)) {
            if(add((MxMeshStructure*)p) != S_OK) {
                Log(LOG_ERROR);
                return E_FAIL;
            }
        } 
        else if(mx::models::vertex::check(p, MxMeshObj::Type::BODY)) {
            if(add((MxMeshBody*)p) != S_OK) {
                Log(LOG_ERROR);
                return E_FAIL;
            }
        }
        else {
            Log(LOG_ERROR) << "Could not determine type of parent";
            return E_FAIL;
        }
    }

    return S_OK;
}


HRESULT MxMesh::removeObj(MxMeshObj *obj) { 
    isDirty = true;

    if(MxMesh_checkStoredObj(obj, this) != S_OK) {
        Log(LOG_ERROR) << "Invalid mesh object passed for remove.";
        return E_FAIL;
    } 

    if(mx::models::vertex::check(obj, MxMeshObj::Type::VERTEX)) {
        MXMESH_OBJINVCHECK(obj, this->vertices);
        this->vertices[obj->objId] = NULL;
        this->vertexIdsAvail.insert(obj->objId);
    } 
    else if(mx::models::vertex::check(obj, MxMeshObj::Type::SURFACE)) {
        MXMESH_OBJINVCHECK(obj, this->surfaces);
        this->surfaces[obj->objId] = NULL;
        this->surfaceIdsAvail.insert(obj->objId);
    } 
    else if(mx::models::vertex::check(obj, MxMeshObj::Type::BODY)) {
        MXMESH_OBJINVCHECK(obj, this->bodies);
        this->bodies[obj->objId] = NULL;
        this->bodyIdsAvail.insert(obj->objId);
    } 
    else if(mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE)) {
        MXMESH_OBJINVCHECK(obj, this->structures);
        this->structures[obj->objId] = NULL;
        this->structureIdsAvail.insert(obj->objId);
    } 
    else {
        Log(LOG_ERROR) << "Mesh object type could not be determined.";
        return E_FAIL;
    }

    obj->objId = -1;
    obj->mesh = NULL;

    for(auto &c : obj->children()) 
        if(removeObj(c) != S_OK) {
            Log(LOG_ERROR);
            return E_FAIL;
        }

    return S_OK;
}

MxMeshVertex *MxMesh::findVertex(const MxVector3f &pos, const float &tol) {

    for(auto &v : vertices)
        if(v->particle()->relativePosition(pos).length() <= tol) 
            return v;

    return NULL;
}

MxMeshVertex *MxMesh::getVertex(const unsigned int &idx) {
    return MXMESH_GETPART(idx, vertices);
}

MxMeshSurface *MxMesh::getSurface(const unsigned int &idx) {
    return MXMESH_GETPART(idx, surfaces);
}

MxMeshBody *MxMesh::getBody(const unsigned int &idx) {
    return MXMESH_GETPART(idx, bodies);
}

MxMeshStructure *MxMesh::getStructure(const unsigned int &idx) {
    return MXMESH_GETPART(idx, structures);
}

template <typename T> 
bool MxMesh_validateInv(const std::vector<T*> &inv) {
    for(auto &o : inv) 
        if(!o->validate()) 
            return false;
    return true;
}

bool MxMesh::validate() {
    if(!MxMesh_validateInv(this->vertices)) 
        return false;
    else if(!MxMesh_validateInv(this->surfaces)) 
        return false;
    else if(!MxMesh_validateInv(this->bodies)) 
        return false;
    else if(!MxMesh_validateInv(this->structures)) 
        return false;
    return true;
}

HRESULT MxMesh::makeDirty() {
    isDirty = true;
    if(_solver) 
        if(_solver->setDirty(true) != S_OK) 
            return E_FAIL;
    return S_OK;
}

bool MxMesh::connected(MxMeshVertex *v1, MxMeshVertex *v2) {
    for(auto &s1 : v1->surfaces) 
        for(auto vitr = s1->vertices.begin() + 1; vitr != s1->vertices.end(); vitr++) 
            if((*vitr == v1 && *(vitr - 1) == v2) || (*vitr == v2 && *(vitr - 1) == v1)) 
                return true;

    return false;
}

bool MxMesh::connected(MxMeshSurface *s1, MxMeshSurface *s2) {
    for(auto &v : s1->parents()) 
        if(v->in(s2)) 
            return true;
    return false;
}

bool MxMesh::connected(MxMeshBody *b1, MxMeshBody *b2) {
    for(auto &s : b1->parents()) 
        if(s->in(b2)) 
            return true;
    return false;
}

HRESULT MxMesh_surfaceOutwardNormal(MxMeshSurface *s, MxMeshBody *b1, MxMeshBody *b2, MxVector3f &onorm) {
    if(b1 && b2) { 
        Log(LOG_ERROR) << "Surface is twice-connected";
        return NULL;
    } 
    else if(b1) {
        onorm = s->getNormal();
    } 
    else if(b2) {
        onorm = -s->getNormal();
    } 
    else {
        onorm = s->getNormal();
    }
    return S_OK;
}

// Mesh editing

HRESULT MxMesh::remove(MxMeshVertex *v) {
    return removeObj(v);
}

HRESULT MxMesh::remove(MxMeshSurface *s) {
    return removeObj(s);
}

HRESULT MxMesh::remove(MxMeshBody *b) {
    return removeObj(b);
}

HRESULT MxMesh::insert(MxMeshVertex *toInsert, MxMeshVertex *v1, MxMeshVertex *v2) {
    MxMeshVertex *v;
    std::vector<MxMeshVertex*>::iterator vitr;

    // Find the common surface(s)
    int nidx;
    MxMeshVertex *vn;
    for(auto &s1 : v1->surfaces) {
        nidx = 0;
        for(vitr = s1->vertices.begin(); vitr != s1->vertices.end(); vitr++) {
            nidx++;
            if(nidx >= s1->vertices.size()) 
                nidx -= s1->vertices.size();
            vn = s1->vertices[nidx];

            if((*vitr == v1 && vn == v2) || (*vitr == v2 && vn == v1)) {
                s1->vertices.insert(vitr + 1 == s1->vertices.end() ? s1->vertices.begin() : vitr + 1, toInsert);
                toInsert->addChild(s1);
                break;
            }
        }
    }

    if(add(toInsert) != S_OK) 
        return E_FAIL;

    if(_solver && _solver->positionChanged() != S_OK) 
        return E_FAIL;

    return S_OK;
}

HRESULT MxMesh::replace(MxMeshVertex *toInsert, MxMeshSurface *toReplace) {
    // For every surface connected to the replaced surface
    //      Gather every vertex connected to the replaced surface
    //      Replace all vertices with the inserted vertex
    // Remove the replaced surface from the mesh
    // Add the inserted vertex to the mesh

    // Gather every contacting surface
    std::vector<MxMeshSurface*> connectedSurfaces;
    for(auto &v : toReplace->vertices) 
        for(auto &s : v->surfaces) 
            if(s != toReplace && std::find(connectedSurfaces.begin(), connectedSurfaces.end(), s) != connectedSurfaces.end()) 
                connectedSurfaces.push_back(s);

    // Disconnect every vertex connected to the replaced surface
    unsigned int lab;
    std::vector<unsigned int> edgeLabels;
    std::vector<MxMeshVertex*> totalToRemove;
    for(auto &s : connectedSurfaces) {
        edgeLabels = s->contiguousEdgeLabels(toReplace);
        std::vector<MxMeshVertex*> toRemove;
        for(unsigned int i = 0; i < edgeLabels.size(); i++) {
            lab = edgeLabels[i];
            if(lab > 0) {
                if(lab > 1) {
                    Log(LOG_ERROR) << "Replacement cannot occur over non-contiguous contacts";
                    return E_FAIL;
                }
                toRemove.push_back(s->vertices[i]);
            }
        }
        
        s->vertices.insert(std::find(s->vertices.begin(), s->vertices.end(), toRemove[0]), toInsert);
        toInsert->addChild(s);
        for(auto &v : toRemove) {
            s->removeParent(v);
            v->removeChild(s);
            totalToRemove.push_back(v);
        }
    }

    // Remove the replaced surface and its vertices
    if(removeObj(toReplace) != S_OK) 
        return E_FAIL;
    for(auto &v : totalToRemove) 
        if(removeObj(v) != S_OK) 
            return E_FAIL;

    // Add the inserted vertex
    if(add(toInsert) != S_OK) 
        return E_FAIL;

    if(_solver && _solver->positionChanged() != S_OK) 
        return E_FAIL;

    return S_OK;
}

MxMeshSurface *MxMesh::replace(MxMeshSurfaceType *toInsert, MxMeshVertex *toReplace, std::vector<float> lenCfs) {
    std::vector<MxMeshVertex*> neighbors = toReplace->neighborVertices();
    if(lenCfs.size() != neighbors.size()) {
        Log(LOG_ERROR) << "Length coefficients are inconsistent with connectivity";
        return NULL;
    } 

    for(auto &cf : lenCfs) 
        if(cf <= 0.f || cf >= 1.f) {
            Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return NULL;
        }

    // Insert new vertices
    MxVector3f pos0 = toReplace->getPosition();
    std::vector<MxMeshVertex*> insertedVertices;
    for(unsigned int i = 0; i < neighbors.size(); i++) {
        float cf = lenCfs[i];
        if(cf <= 0.f || cf >= 1.f) {
            Log(LOG_ERROR) << "Length coefficients must be in (0, 1)";
            return NULL;
        }

        MxMeshVertex *v = neighbors[i];
        MxVector3f pos1 = v->getPosition();
        MxVector3f pos = pos0 + (pos1 - pos0) * cf;
        MxMeshParticleType *ptype = MxMeshParticleType_get();
        MxParticleHandle *ph = (*ptype)(&pos);
        MxMeshVertex *vInserted = new MxMeshVertex(ph->id);
        if(insert(vInserted, toReplace, v) != S_OK) 
            return NULL;
        insertedVertices.push_back(vInserted);
    }

    // Disconnect replaced vertex from all surfaces
    std::vector<MxMeshSurface*> toReplaceSurfaces(toReplace->surfaces.begin(), toReplace->surfaces.end());
    for(auto &s : toReplaceSurfaces) {
        s->removeParent(toReplace);
        toReplace->removeChild(s);
    }

    // Create new surface; its constructor should handle internal connections
    MxMeshSurface *inserted = (*toInsert)(insertedVertices);

    // Remove replaced vertex from the mesh and add inserted surface to the mesh
    removeObj(toReplace);
    add(inserted);

    if(_solver) 
        _solver->positionChanged();

    return inserted;
}

HRESULT MxMesh::merge(MxMeshVertex *toKeep, MxMeshVertex *toRemove, const float &lenCf) {
    // Impose that vertices with shared surfaces must be adjacent
    auto sharedSurfaces = toKeep->sharedSurfaces(toRemove);
    if(sharedSurfaces.size() == 0) {
        Log(LOG_ERROR) << "Vertices must be adjacent on shared surfaces";
        return E_FAIL;
    }
    auto s = sharedSurfaces[0];
    MxMeshVertex *vt;
    for(unsigned int i = 0; i < s->vertices.size(); i++) {
        auto v = s->vertices[i];
        vt = v == toKeep ? toRemove : (v == toRemove ? toKeep : NULL);

        if(vt) {
            if(vt != s->vertices[i + 1 >= s->vertices.size() ? i + 1 - s->vertices.size() : i + 1]) {
                Log(LOG_ERROR) << "Vertices with shared surfaces must be adjacent";
                return E_FAIL;
            } 
            else 
                break;
        }
    }

    // Disconnect and remove vertex
    auto childrenToRemove = toRemove->children();
    for(auto &c : childrenToRemove) {
        if(toRemove->removeChild(c) != S_OK) 
            return E_FAIL;
        if(c->removeParent(toRemove) != S_OK) 
            return E_FAIL;
    }
    if(remove(toRemove) != S_OK) 
        return E_FAIL;
    
    // Set new position
    const MxVector3f posToKeep = toKeep->getPosition();
    const MxVector3f newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
    if(toKeep->setPosition(newPos) != S_OK) 
        return E_FAIL;

    if(_solver && _solver->positionChanged() != S_OK) 
        return E_FAIL;

    return S_OK;
}

HRESULT MxMesh::merge(MxMeshSurface *toKeep, MxMeshSurface *toRemove, const std::vector<float> &lenCfs) {
    if(toKeep->vertices.size() != toRemove->vertices.size()) {
        Log(LOG_ERROR) << "Surfaces must have the same number of vertices to merge";
        return E_FAIL;
    }

    // Find vertices that are not shared
    std::vector<MxMeshVertex*> toKeepExcl;
    for(auto &v : toKeep->vertices) 
        if(!v->in(toRemove)) 
            toKeepExcl.push_back(v);

    // Ensure sufficient length cofficients
    std::vector<float> _lenCfs = lenCfs;
    if(_lenCfs.size() < toKeepExcl.size()) {
        Log(LOG_DEBUG) << "Insufficient provided length coefficients. Assuming 0.5";
        for(unsigned int i = _lenCfs.size(); i < toKeepExcl.size(); i++) 
            _lenCfs.push_back(0.5);
    }

    // Match vertex order of removed surface to kept surface by nearest distance
    std::vector<MxMeshVertex*> toRemoveOrdered;
    for(auto &kv : toKeepExcl) {
        MxMeshVertex *mv = NULL;
        MxVector3f kp = kv->getPosition();
        float bestDist = 0.f;
        for(auto &rv : toRemove->vertices) {
            float dist = (rv->getPosition() - kp).length();
            if((!mv || dist < bestDist) && std::find(toRemoveOrdered.begin(), toRemoveOrdered.end(), rv) == toRemoveOrdered.end()) {
                bestDist = dist;
                mv = rv;
            }
        }
        if(!mv) {
            Log(LOG_ERROR) << "Could not match surface vertices";
            return E_FAIL;
        }
        toRemoveOrdered.push_back(mv);
    }

    // Replace vertices in neighboring surfaces
    for(unsigned int i = 0; i < toKeepExcl.size(); i++) {
        MxMeshVertex *rv = toRemoveOrdered[i];
        MxMeshVertex *kv = toKeepExcl[i];
        std::vector<MxMeshSurface*> rvSurfaces = rv->surfaces;
        for(auto &s : rvSurfaces) 
            if(s != toRemove) {
                if(std::find(s->vertices.begin(), s->vertices.end(), rv) == s->vertices.end()) {
                    Log(LOG_ERROR) << "Something went wrong during surface merge";
                    return E_FAIL;
                }
                std::replace(s->vertices.begin(), s->vertices.end(), rv, kv);
                kv->surfaces.push_back(s);
            }
    }

    // Replace surface in child bodies
    for(auto &b : toRemove->getBodies()) {
        if(!toKeep->in(b)) {
            b->addParent(toKeep);
            toKeep->addChild(b);
        }
        b->removeParent(toRemove);
        toRemove->removeChild(b);
    }

    // Detach removed vertices
    for(auto &v : toRemoveOrdered) {
        v->surfaces.clear();
        toRemove->removeParent(v);
    }

    // Move kept vertices by length coefficients
    for(unsigned int i = 0; i < toKeepExcl.size(); i++) {
        MxMeshVertex *v = toKeepExcl[i];
        MxVector3f posToKeep = v->getPosition();
        MxVector3f newPos = posToKeep + (toRemoveOrdered[i]->getPosition() - posToKeep) * _lenCfs[i];
        if(v->setPosition(newPos) != S_OK) 
            return E_FAIL;
    }
    
    // Remove surface and vertices that are not shared
    if(remove(toRemove) != S_OK) 
        return E_FAIL;
    for(auto &v : toRemoveOrdered) 
        if(remove(v) != S_OK) 
            return E_FAIL;

    if(_solver && _solver->positionChanged() != S_OK) 
        return E_FAIL;

    return S_OK;
}

MxMeshSurface *MxMesh::extend(MxMeshSurface *base, const unsigned int &vertIdxStart, const MxVector3f &pos) {
    // Validate indices
    if(vertIdxStart >= base->vertices.size()) {
        Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << base->vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    MxMeshVertex *v0 = base->vertices[vertIdxStart];
    MxMeshVertex *v1 = base->vertices[vertIdxStart == base->vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertex at specified position
    MxMeshSurfaceType *stype = base->type();
    MxMeshParticleType *ptype = MxMeshParticleType_get();
    MxVector3f _pos = pos;
    MxParticleHandle *ph = (*ptype)(&_pos);
    MxMeshVertex *vert = new MxMeshVertex(ph->id);

    // Construct new surface, add new parts and return
    MxMeshSurface *s = (*stype)({v0, v1, vert});
    add(s);

    if(_solver) 
        _solver->positionChanged();

    return s;
}

MxMeshSurface *MxMesh::extrude(MxMeshSurface *base, const unsigned int &vertIdxStart, const float &normLen) {
    // Validate indices
    if(vertIdxStart >= base->vertices.size()) {
        Log(LOG_ERROR) << "Invalid vertex indices (" << vertIdxStart << ", " << base->vertices.size() << ")";
        return NULL;
    }

    // Get base vertices
    MxMeshVertex *v0 = base->vertices[vertIdxStart];
    MxMeshVertex *v1 = base->vertices[vertIdxStart == base->vertices.size() - 1 ? 0 : vertIdxStart + 1];

    // Construct new vertices
    MxVector3f disp = base->normal * normLen;
    MxVector3f pos2 = v0->getPosition() + disp;
    MxVector3f pos3 = v1->getPosition() + disp;
    MxMeshParticleType *ptype = MxMeshParticleType_get();
    MxParticleHandle *p2 = (*ptype)(&pos2);
    MxParticleHandle *p3 = (*ptype)(&pos3);
    MxMeshVertex *v2 = new MxMeshVertex(p2->id);
    MxMeshVertex *v3 = new MxMeshVertex(p3->id);

    // Construct new surface, add new parts and return
    MxMeshSurfaceType *stype = base->type();
    MxMeshSurface *s = (*stype)({v0, v1, v2, v3});
    add(s);

    if(_solver) 
        _solver->positionChanged();

    return s;
}

MxMeshBody *MxMesh::extend(MxMeshSurface *base, MxMeshBodyType *btype, const MxVector3f &pos) {
    // For every pair of vertices, construct a surface with a new vertex at the given position
    MxMeshVertex *vNew = new MxMeshVertex(pos);
    MxMeshSurfaceType *stype = base->type();
    std::vector<MxMeshSurface*> surfaces(1, base);
    for(unsigned int i = 0; i < base->vertices.size(); i++) {
        // Get base vertices
        MxMeshVertex *v0 = base->vertices[i];
        MxMeshVertex *v1 = base->vertices[i == base->vertices.size() - 1 ? 0 : i + 1];

        MxMeshSurface *s = (*stype)({v0, v1, vNew});
        if(!s) 
            return NULL;
        surfaces.push_back(s);
    }

    // Construct a body from the surfaces
    MxMeshBody *b = (*btype)(surfaces);
    if(!b) 
        return NULL;

    // Add new parts and return
    add(b);

    if(_solver) 
        _solver->positionChanged();

    return b;
}

MxMeshBody *MxMesh::extrude(MxMeshSurface *base, MxMeshBodyType *btype, const float &normLen) {
    unsigned int i, j;
    MxVector3f normal;

    // Only permit if the surface has an available slot
    base->refreshBodies();
    if(MxMesh_surfaceOutwardNormal(base, base->b1, base->b2, normal) != S_OK) 
        return NULL;

    std::vector<MxMeshVertex*> newVertices(base->vertices.size(), 0);
    MxMeshSurfaceType *stype = base->type();
    MxMeshParticleType *ptype = MxMeshParticleType_get();
    MxVector3f disp = normal * normLen;

    for(i = 0; i < base->vertices.size(); i++) {
        MxVector3f pos = base->vertices[i]->getPosition() + disp;
        MxParticleHandle *ph = (*ptype)(&pos);
        newVertices[i] = new MxMeshVertex(ph->id);
    }

    std::vector<MxMeshSurface*> newSurfaces;
    for(i = 0; i < base->vertices.size(); i++) {
        j = i + 1 >= base->vertices.size() ? i + 1 - base->vertices.size() : i + 1;
        MxMeshSurface *s = (*stype)({
            base->vertices[i], 
            base->vertices[j], 
            newVertices[j], 
            newVertices[i]
        });
        if(!s) 
            return NULL;
        newSurfaces.push_back(s);
    }
    newSurfaces.push_back(base);
    newSurfaces.push_back((*stype)(newVertices));

    MxMeshBody *b = (*btype)(newSurfaces);
    if(!b) 
        return NULL;
    add(b);

    if(_solver) 
        _solver->positionChanged();

    return b;
}
