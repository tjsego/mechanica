/**
 * @file Mx3DFMeshData.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Mechanica 3D data format mesh data
 * @date 2021-12-15
 * 
 */

#include <MxUtil.h>

#include "Mx3DFVertexData.h"
#include "Mx3DFEdgeData.h"
#include "Mx3DFFaceData.h"
#include "Mx3DFMeshData.h"


std::vector<Mx3DFVertexData*> Mx3DFMeshData::getVertices() {
    std::vector<Mx3DFVertexData*> result;
    for(auto e : this->faces) {
        auto v = e->getVertices();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

std::vector<Mx3DFEdgeData*> Mx3DFMeshData::getEdges() {
    std::vector<Mx3DFEdgeData*> result;
    for(auto e : this->faces) {
        auto v = e->getEdges();
        result.insert(result.end(), v.begin(), v.end());
    }
    return unique(result);
}

std::vector<Mx3DFFaceData*> Mx3DFMeshData::getFaces() {
    return this->faces;
}

unsigned int Mx3DFMeshData::getNumVertices() {
    return this->getVertices().size();
}

unsigned int Mx3DFMeshData::getNumEdges() {
    return this->getEdges().size();
}

unsigned int Mx3DFMeshData::getNumFaces() {
    return this->faces.size();
}

bool Mx3DFMeshData::has(Mx3DFVertexData *v) {
    for(auto f : this->faces) 
        if(f->has(v)) 
            return true;
    return false;
}

bool Mx3DFMeshData::has(Mx3DFEdgeData *e) {
    for(auto f : this->faces) 
        if(f->has(e)) 
            return true;
    return false;
}

bool Mx3DFMeshData::has(Mx3DFFaceData *f) {
    auto itr = std::find(this->faces.begin(), this->faces.end(), f);
    return itr != std::end(this->faces);
}

bool Mx3DFMeshData::in(Mx3DFStructure *s) {
    return s == this->structure;
}

MxVector3f Mx3DFMeshData::getCentroid() { 
    auto vertices = this->getVertices();
    auto numV = vertices.size();

    if(numV == 0) 
        mx_error(E_FAIL, "No vertices");

    MxVector3f result = {0.f, 0.f, 0.f};

    for(unsigned int i = 0; i < numV; i++) 
        result += vertices[i]->position;

    result /= numV;
    return result;
}

HRESULT Mx3DFMeshData::translate(const MxVector3f &displacement) {
    for(auto v : this->getVertices()) 
        v->position += displacement;
    
    return S_OK;
}

HRESULT Mx3DFMeshData::translateTo(const MxVector3f &position) {
    return this->translate(position - this->getCentroid());
}

HRESULT Mx3DFMeshData::rotateAt(const MxMatrix3f &rotMat, const MxVector3f &rotPt) {
    MxMatrix4f t = Magnum::Matrix4::translation(rotPt) * Magnum::Matrix4::from(rotMat, MxVector3f(0.f)) * Magnum::Matrix4::translation(rotPt * -1.f);

    for(auto v : this->getVertices()) { 
        MxVector4f p = {v->position.x(), v->position.y(), v->position.z(), 1.f};
        v->position = (t * p).xyz();
    }

    return S_OK;
}

HRESULT Mx3DFMeshData::rotate(const MxMatrix3f &rotMat) {
    return this->rotateAt(rotMat, this->getCentroid());
}

HRESULT Mx3DFMeshData::scaleFrom(const MxVector3f &scales, const MxVector3f &scalePt) {
    if(scales[0] <= 0 || scales[1] <= 0 || scales[2] <= 0) 
        mx_error(E_FAIL, "Invalid non-positive scale");

    MxMatrix4f t = Magnum::Matrix4::translation(scalePt) * Magnum::Matrix4::scaling(scales) * Magnum::Matrix4::translation(scalePt * -1.f);

    for(auto v : this->getVertices()) { 
        MxVector4f p = {v->position.x(), v->position.y(), v->position.z(), 1.f};
        v->position = (t * p).xyz();
    }

    return S_OK;
}

HRESULT Mx3DFMeshData::scaleFrom(const float &scale, const MxVector3f &scalePt) {
    return this->scaleFrom(MxVector3f(scale), scalePt);
}

HRESULT Mx3DFMeshData::scale(const MxVector3f &scales) {
    return this->scaleFrom(scales, this->getCentroid());
}

HRESULT Mx3DFMeshData::scale(const float &scale) {
    return this->scale(MxVector3f(scale));
}
