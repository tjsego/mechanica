/**
 * @file MxMeshStructure.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica mesh structure class
 * @date 2022-26-04
 * 
 */

#include "MxMeshStructure.h"

#include "MxMeshBody.h"
#include "MxMeshSolver.h"

#include <MxLogger.h>
#include <MxUtil.h>


std::vector<MxMeshObj*> MxMeshStructure::parents() {
    std::vector<MxMeshObj*> result(structures_parent.size() + bodies.size(), 0);
    for(unsigned int i = 0; i < structures_parent.size(); i++) 
        result[i] = static_cast<MxMeshObj*>(structures_parent[i]);
    for(unsigned int i = 0; i < bodies.size(); i++) 
        result[structures_parent.size() + i] = static_cast<MxMeshObj*>(bodies[i]);
    return result;
}

HRESULT MxMeshStructure::addChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshStructure *s = (MxMeshStructure*)obj;
    if(std::find(structures_child.begin(), structures_child.end(), s) != structures_child.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    structures_child.push_back(s);
    return S_OK;
}

HRESULT MxMeshStructure::addParent(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE) && !mx::models::vertex::check(obj, MxMeshObj::Type::BODY)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    if(obj->objType() == MxMeshObj::Type::BODY) {
        MxMeshBody *b = (MxMeshBody*)obj;
        if(std::find(bodies.begin(), bodies.end(), b) != bodies.end()) {
            Log(LOG_ERROR);
            return E_FAIL;
        }
        bodies.push_back(b);
    } 
    else {
        MxMeshStructure *s = (MxMeshStructure*)obj;
        if(std::find(structures_parent.begin(), structures_parent.end(), s) != structures_parent.end()) {
            Log(LOG_ERROR);
            return E_FAIL;
        }
        structures_parent.push_back(s);
    }
    
    return S_OK;
}

HRESULT MxMeshStructure::removeChild(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    MxMeshStructure *s = (MxMeshStructure*)obj;
    auto itr = std::find(structures_child.begin(), structures_child.end(), s);
    if(itr == structures_child.end()) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    structures_child.erase(itr);
    return S_OK;
}

HRESULT MxMeshStructure::removeParent(MxMeshObj *obj) {
    if(!mx::models::vertex::check(obj, MxMeshObj::Type::STRUCTURE) && !mx::models::vertex::check(obj, MxMeshObj::Type::BODY)) {
        Log(LOG_ERROR);
        return E_FAIL;
    }

    if(obj->objType() == MxMeshObj::Type::BODY) {
        MxMeshBody *b = (MxMeshBody*)obj;
        auto itr = std::find(bodies.begin(), bodies.end(), b);
        if(itr != bodies.end()) {
            Log(LOG_ERROR);
            return E_FAIL;
        }
        bodies.erase(itr);
    } 
    else {
        MxMeshStructure *s = (MxMeshStructure*)obj;
        auto itr = std::find(structures_parent.begin(), structures_parent.end(), s);
        if(itr != structures_parent.end()) {
            Log(LOG_ERROR);
            return E_FAIL;
        }
        structures_parent.erase(itr);
    }
    
    return S_OK;
}

MxMeshStructureType *MxMeshStructure::type() {
    MxMeshSolver *solver = MxMeshSolver::get();
    if(!solver) 
        return NULL;
    return solver->getStructureType(typeId);
}

std::vector<MxMeshBody*> MxMeshStructure::getBodies() {
    std::vector<MxMeshBody*> result(bodies.begin(), bodies.end());
    for(auto &sp : structures_parent) 
        for(auto &b : sp->getBodies()) 
            result.push_back(b);
    return unique(result);
}

std::vector<MxMeshSurface*> MxMeshStructure::getSurfaces() {
    std::vector<MxMeshSurface*> result;
    for(auto &b : bodies) 
        for(auto &s : b->getSurfaces()) 
            result.push_back(s);
    for(auto &sp : structures_parent) 
        for(auto &s : sp->getSurfaces()) 
            result.push_back(s);
    return unique(result);
}

std::vector<MxMeshVertex*> MxMeshStructure::getVertices() {
    std::vector<MxMeshVertex*> result;
    for(auto &b : bodies) 
        for(auto &v : b->getVertices()) 
            result.push_back(v);
    for(auto &sp : structures_parent) 
        for(auto &v : sp->getVertices()) 
            result.push_back(v);
    return unique(result);
}
