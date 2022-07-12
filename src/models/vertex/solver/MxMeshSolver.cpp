/**
 * @file MxMeshSolver.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh solver
 * @date 2022-26-04
 * 
 */

// todo: parallelize execution

#include "MxMeshSolver.h"

#include "MxMeshObj.h"
#include "MxMeshRenderer.h"

#include <engine.h>
#include <MxUtil.h>
#include <MxLogger.h>


static MxMeshSolver *_solver = NULL;


#define MXMESHSOLVER_CHECKINIT { if(!_solver) return E_FAIL; }

HRESULT MxMeshSolver::init() {
    if(_solver != NULL) 
        return S_OK;

    _solver = new MxMeshSolver();
    _solver->_bufferSize = 1;
    _solver->_forces = (float*)malloc(3 * sizeof(float));
    _solver->registerEngine();

    // Launches and registers renderer
    MxMeshRenderer::get();

    return S_OK;
}

MxMeshSolver *MxMeshSolver::get() { 
    if(_solver == NULL) 
        if(init() != S_OK) 
            return NULL;
    return _solver;
}

HRESULT MxMeshSolver::compact() { 
    MXMESHSOLVER_CHECKINIT

    if(_solver->_bufferSize > 1) {
        free(_forces);
        _bufferSize = 1;
        _forces = (float*)malloc(3 * sizeof(float));
    }

    return S_OK;
}

MxMesh *MxMeshSolver::newMesh() {
    MxMesh *mesh = new MxMesh();
    if(loadMesh(mesh) != S_OK) 
        return NULL;
    return mesh;
}

HRESULT MxMeshSolver::loadMesh(MxMesh *mesh) {
    for(auto &m : meshes) 
        if(m == mesh) 
            return E_FAIL;
    meshes.push_back(mesh);
    mesh->_solver = this;
    mesh->isDirty = true;
    _isDirty = true;
    return S_OK;
}

HRESULT MxMeshSolver::unloadMesh(MxMesh *mesh) {
    for(auto itr = meshes.begin(); itr != meshes.end(); itr++) {
        if(*itr == mesh) {
            meshes.erase(itr);
            _isDirty = true;
            (*itr)->_solver = NULL;
            return S_OK;
        }
    }
    return E_FAIL;
}

HRESULT MxMeshSolver::registerType(MxMeshBodyType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;
    
    _type->id = _bodyTypes.size();
    _bodyTypes.push_back(_type);

    return S_OK;
}

HRESULT MxMeshSolver::registerType(MxMeshSurfaceType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;

    _type->id = _surfaceTypes.size();
    if(!_type->style) {
        auto colors = MxColor3_Names();
        auto c = colors[(_surfaceTypes.size() - 1) % colors.size()];
        _type->style = new MxStyle(c);
    }
    _surfaceTypes.push_back(_type);

    return S_OK;
}

MxMeshStructureType *MxMeshSolver::getStructureType(const unsigned int &typeId) {
    if(typeId >= _structureTypes.size()) 
        return NULL;
    return _structureTypes[typeId];
}

MxMeshBodyType *MxMeshSolver::getBodyType(const unsigned int &typeId) {
    if(typeId >= _bodyTypes.size()) 
        return NULL;
    return _bodyTypes[typeId];
}

MxMeshSurfaceType *MxMeshSolver::getSurfaceType(const unsigned int &typeId) {
    if(typeId >= _surfaceTypes.size()) 
        return NULL;
    return _surfaceTypes[typeId];
}

template <typename T> 
void MxMesh_actRecursive(MxMeshObj *vertex, T *source, float *f) {
    for(auto &a : source->type()->actors) 
        a->force(source, vertex, f);
    for(auto &c : source->children()) 
        MxMesh_actRecursive(vertex, (T*)c, f);
}

HRESULT MxMeshSolver::positionChanged() {

    unsigned int i;
    _surfaceVertices = 0;
    _totalVertices = 0;

    for(auto &m : meshes) {
        for(i = 0; i < m->sizeVertices(); i++) {
            MxMeshVertex *v = m->getVertex(i);
            if(v) 
                v->positionChanged();
        }
        _totalVertices += m->numVertices();

        for(i = 0; i < m->sizeSurfaces(); i++) {
            MxMeshSurface *s = m->getSurface(i);
            if(s) {
                s->positionChanged();
                _surfaceVertices += s->parents().size();
            }
        }

        for(i = 0; i < m->sizeBodies(); i++) {
            MxMeshBody *b = m->getBody(i);
            if(b) 
                b->positionChanged();
        }
        
        for(i = 0; i < m->sizeVertices(); i++) {
            MxMeshVertex *v = m->getVertex(i);
            if(v) 
                v->updateProperties();
        }

        m->isDirty = false;
    }

    _isDirty = false;

    return S_OK;
}

HRESULT MxMeshSolver::update(const bool &_force) {
    if(!isDirty() || _force) 
        return S_OK;
    
    positionChanged();
    return S_OK;
}

HRESULT MxMeshSolver::preStepStart() { 
    MXMESHSOLVER_CHECKINIT

    MxMeshLogger::clear();

    unsigned int i, j, k;
    MxMesh *m;
    MxMeshVertex *v;
    MxMeshSurface *s;
    MxMeshBody *b;

    _surfaceVertices = 0;
    _totalVertices = 0;

    for(i = 0; i < meshes.size(); i++) {
        j = meshes[i]->sizeVertices();
        _totalVertices += j;
    }

    if(_totalVertices > _bufferSize) {
        free(_solver->_forces);
        _bufferSize = _totalVertices;
        _solver->_forces = (float*)malloc(3 * sizeof(float) * _bufferSize);
    }
    memset(_solver->_forces, 0.f, 3 * sizeof(float) * _bufferSize);

    for(i = 0, j = 0; i < meshes.size(); i++) { 
        m = meshes[i];
        for(k = 0; k < m->sizeVertices(); k++, j++) {
            v = m->getVertex(k);
            
            if(!v) 
                continue;
            
            float *buff = &_forces[j * 3];

            // Surfaces
            for(auto &s : v->getSurfaces()) {
                for(auto &a : s->type()->actors) 
                    a->force(s, v, buff);
                
                for(auto &a : s->actors) 
                    a->force(s, v, buff);

                _surfaceVertices++;
            }

            // Bodies
            for(auto &b : v->getBodies()) {
                for(auto &a : b->type()->actors) 
                    a->force(b, v, buff);

                for(auto &a : b->actors) 
                    a->force(b, v, buff);
            }

            // Structures
            for(auto &st : v->getStructures()) {
                for(auto &a : st->type()->actors) 
                    a->force(st, v, buff);

                for(auto &a : st->actors) 
                    a->force(st, v, buff);
            }
        }
    }

    return S_OK;
}

HRESULT MxMeshSolver::preStepJoin() {
    unsigned int i, j;
    float *buff;
    MxMesh *m;
    MxParticle *p;

    for(i = 0, j = 0; i < meshes.size(); i++) { 
        m = meshes[i];
        for(auto &v : m->vertices) {
            if(!v) {
                j++;
                continue;
            }

            p = v->particle()->part();
            buff = &_forces[j * 3];
            p->f[0] += buff[0];
            p->f[1] += buff[1];
            p->f[2] += buff[2];
            j++;
        }
    }

    return S_OK;
}

HRESULT MxMeshSolver::postStepStart() {
    setDirty(true);
    return positionChanged();
}

HRESULT MxMeshSolver::postStepJoin() {
    return S_OK;
}

HRESULT MxMeshSolver::log(MxMesh *mesh, const MxMeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MxMeshObj::Type> &objTypes) {
    int meshID = -1;
    for(int i = 0; i < meshes.size(); i++) 
        if(meshes[i] == mesh) {
            meshID = i;
            break;
        }

    if(meshID < 0) {
        Log(LOG_ERROR) << "Mesh not in solved";
        return E_FAIL;
    }

    MxMeshLogEvent event;
    event.meshID = meshID;
    event.type = type;
    event.objIDs = objIDs;
    event.objTypes = objTypes;
    return MxMeshLogger::log(event);
}

bool MxMeshSolver::isDirty() {
    if(_isDirty) 
        return true;
    bool result = false;
    for(auto &m : meshes) 
        result |= m->isDirty;
    return result;
}

HRESULT MxMeshSolver::setDirty(const bool &_dirty) {
    _isDirty = _dirty;
    for(auto &m : meshes) 
        m->isDirty = _dirty;
    return S_OK;
}
