/**
 * @file Mx3DFVertexData.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format vertex data
 * @date 2021-12-15
 * 
 */

#include <MxUtil.h>

#include "Mx3DFVertexData.h"
#include "Mx3DFEdgeData.h"
#include "Mx3DFFaceData.h"
#include "Mx3DFMeshData.h"


Mx3DFVertexData::Mx3DFVertexData(const MxVector3f &_position, Mx3DFStructure *_structure) : 
    structure{_structure}, 
    position{_position}
{}

std::vector<Mx3DFEdgeData*> Mx3DFVertexData::getEdges() {
    return edges;
}

std::vector<Mx3DFFaceData*> Mx3DFVertexData::getFaces() {
    std::vector<Mx3DFFaceData*> result;
    for(auto e : this->edges) {
        auto v = e->getFaces();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

std::vector<Mx3DFMeshData*> Mx3DFVertexData::getMeshes() {
    std::vector<Mx3DFMeshData*> result;
    for(auto e : this->edges) {
        auto v = e->getMeshes();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

unsigned int Mx3DFVertexData::getNumEdges() {
    return this->edges.size();
}

unsigned int Mx3DFVertexData::getNumFaces() {
    return this->getFaces().size();
}

unsigned int Mx3DFVertexData::getNumMeshes() {
    return this->getMeshes().size();
}

bool Mx3DFVertexData::in(Mx3DFEdgeData *e) {
    auto itr = std::find(this->edges.begin(), this->edges.end(), e);
    return itr != std::end(this->edges);
}

bool Mx3DFVertexData::in(Mx3DFFaceData *f) {
    for(auto e : this->edges) 
        if(e->in(f)) 
            return true;
    return false;
}

bool Mx3DFVertexData::in(Mx3DFMeshData *m) {
    for(auto e : this->edges) 
        if(e->in(m)) 
            return true;
    return false;
}

bool Mx3DFVertexData::in(Mx3DFStructure *s) {
    return s == this->structure;
}
