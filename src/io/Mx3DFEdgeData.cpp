/**
 * @file Mx3DFEdgeData.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format edge data
 * @date 2021-12-15
 * 
 */

#include <MxUtil.h>

#include "Mx3DFVertexData.h"
#include "Mx3DFEdgeData.h"
#include "Mx3DFFaceData.h"
#include "Mx3DFMeshData.h"


Mx3DFEdgeData::Mx3DFEdgeData(Mx3DFVertexData *_va, Mx3DFVertexData *_vb) : 
    va{_va}, 
    vb{_vb}
{
    if(this->va == NULL || this->vb == NULL) 
        mx_error(E_FAIL, "Invalid vertex (NULL)");

    this->va->edges.push_back(this);
    this->vb->edges.push_back(this);
}

std::vector<Mx3DFVertexData*> Mx3DFEdgeData::getVertices() {
    return {this->va, this->vb};
}

std::vector<Mx3DFFaceData*> Mx3DFEdgeData::getFaces() {
    return this->faces;
}

std::vector<Mx3DFMeshData*> Mx3DFEdgeData::getMeshes() {
    std::vector<Mx3DFMeshData*> result;
    for(auto e : this->faces) {
        auto v = e->getMeshes();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

unsigned int Mx3DFEdgeData::getNumVertices() {
    return 2;
}

unsigned int Mx3DFEdgeData::getNumFaces() {
    return this->faces.size();
}

unsigned int Mx3DFEdgeData::getNumMeshes() {
    return this->getMeshes().size();
}

bool Mx3DFEdgeData::has(Mx3DFVertexData *v) {
    return v == this->va || v == this->vb;
}

bool Mx3DFEdgeData::in(Mx3DFFaceData *f) {
    auto itr = std::find(this->faces.begin(), this->faces.end(), f);
    return itr != std::end(this->faces);
}

bool Mx3DFEdgeData::in(Mx3DFMeshData *m) {
    for(auto f : this->faces) 
        if(f->in(m)) 
            return true;
    return false;
}

bool Mx3DFEdgeData::in(Mx3DFStructure *s) {
    return s == this->structure;
}
