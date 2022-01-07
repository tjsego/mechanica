/**
 * @file Mx3DFFaceData.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format face data
 * @date 2021-12-15
 * 
 */

#include <MxUtil.h>

#include "Mx3DFVertexData.h"
#include "Mx3DFEdgeData.h"
#include "Mx3DFFaceData.h"
#include "Mx3DFMeshData.h"


std::vector<Mx3DFVertexData*> Mx3DFFaceData::getVertices() {
    std::vector<Mx3DFVertexData*> result;
    for(auto e : this->edges) {
        auto v = e->getVertices();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

std::vector<Mx3DFEdgeData*> Mx3DFFaceData::getEdges() {
    return this->edges;
}

std::vector<Mx3DFMeshData*> Mx3DFFaceData::getMeshes() {
    return this->meshes;
}

unsigned int Mx3DFFaceData::getNumVertices() {
    return this->getVertices().size();
}

unsigned int Mx3DFFaceData::getNumEdges() {
    return this->edges.size();
}

unsigned int Mx3DFFaceData::getNumMeshes() {
    return this->meshes.size();
}

bool Mx3DFFaceData::has(Mx3DFVertexData *v) {
    for(auto e : this->edges) 
        if(e->has(v)) 
            return true;
    return false;
}

bool Mx3DFFaceData::has(Mx3DFEdgeData *e) {
    auto itr = std::find(this->edges.begin(), this->edges.end(), e);
    return itr != std::end(this->edges);
}

bool Mx3DFFaceData::in(Mx3DFMeshData *m) {
    auto itr = std::find(this->meshes.begin(), this->meshes.end(), m);
    return itr != std::end(this->meshes);
}

bool Mx3DFFaceData::in(Mx3DFStructure *s) {
    return s == this->structure;
}
