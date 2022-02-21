/*
 * MxCutPlane.cpp
 *
 *  Created on: Mar 26, 2021
 *      Author: andy
 */

#include "MxClipPlane.hpp"
#include <Magnum/Math/Distance.h>
#include <MxSystem.h>
#include <rendering/MxUniverseRenderer.h>
#include <mx_error.h>
#include <MxLogger.h>

MxClipPlanes _clipPlanesObj;

MxClipPlane::MxClipPlane(int i) : index(i) {}

MxVector4f MxPlaneEquation(const MxVector3f &normal, const MxVector3f &point) {
    return Magnum::Math::planeEquation(normal, point);
}

HRESULT MxPlaneEquation(const MxVector3f &normal, const MxVector3f &point, float *result) {
    auto r = MxPlaneEquation(normal, point);
    for(unsigned int i = 0; i < r.Size; ++i) {
        result[i] = r[i];
    }
    return S_OK;
}

std::tuple<MxVector3f, MxVector3f> MxPlaneEquation(const MxVector4f &planeEq) {
    MxVector3f normal = planeEq.xyz();
    MxVector3f point;
    int i, j, k;

    if(normal[0] != 0.f) {      i = 1; j = 2; k = 0;}
    else if(normal[1] != 0.f) { i = 0; j = 2; k = 1;}
    else {                      i = 0; j = 1; k = 2;}

    point[i] = point[j] = 0.f;
    point[k] = - planeEq.w() / normal[k];

    return std::make_tuple(normal, point);
}

std::vector<MxVector4f> MxParsePlaneEquation(const std::vector<std::tuple<MxVector3f, MxVector3f> > &clipPlanes) {
    std::vector<MxVector4f> result(clipPlanes.size());
    MxVector3f point;
    MxVector3f normal;

    for(unsigned int i = 0; i < clipPlanes.size(); ++i) {
        std::tie(point, normal) = clipPlanes[i];
        result[i] = MxPlaneEquation(normal, point);
    }
    return result;
}

HRESULT MxParsePlaneEquation(const std::vector<std::tuple<MxVector3f, MxVector3f> > &clipPlanes, MxVector4f *result) {
    unsigned int i = 0;
    for(auto pe : MxParsePlaneEquation(clipPlanes)) {
        result[i] = pe;
        i += 1;
    }
    return S_OK;
}

MxVector3f MxClipPlane::getPoint() {
    auto eq = MxSystem::getRenderer()->getClipPlaneEquation(this->index);
    return std::get<1>(MxPlaneEquation(eq));
}

MxVector3f MxClipPlane::getNormal() {
    auto eq = MxSystem::getRenderer()->getClipPlaneEquation(this->index);
    return std::get<0>(MxPlaneEquation(eq));
}

MxVector4f MxClipPlane::getEquation() {
    return MxSystem::getRenderer()->getClipPlaneEquation(this->index);
}

HRESULT MxClipPlane::setEquation(const MxVector4f &pe) {
    MxSystem::getRenderer()->setClipPlaneEquation(this->index, pe);
    return S_OK;
}

HRESULT MxClipPlane::setEquation(const MxVector3f &point, const MxVector3f &normal) {
    return this->setEquation(MxPlaneEquation(normal, point));
}

HRESULT MxClipPlane::destroy() {
    if(this->index < 0) {
        Log(LOG_CRITICAL) << "Clip plane no longer valid";
        return E_FAIL;
    }

    MxSystem::getRenderer()->removeClipPlaneEquation(this->index);
    this->index = -1;
    return S_OK;
}

int MxClipPlanes::len() {
    return MxSystem::getRenderer()->clipPlaneCount();
}

const MxVector4f &MxClipPlanes::getClipPlaneEquation(const unsigned int &index) {
    try {
        MxUniverseRenderer *renderer = MxSystem::getRenderer();
        
        if(index > renderer->clipPlaneCount()) mx_exp(std::range_error("index out of bounds"));
        return MxVector4f::from(renderer->getClipPlaneEquation(index).data());
    }
    catch(const std::exception &e) {
        mx_error(E_FAIL, e.what());
        MxVector4f *result = new MxVector4f();
        return *result;
    }
}

HRESULT MxClipPlanes::setClipPlaneEquation(const unsigned int &index, const MxVector4f &pe) {
    try {
        MxUniverseRenderer *renderer = MxSystem::getRenderer();
        if(index > renderer->clipPlaneCount()) mx_exp(std::range_error("index out of bounds"));
        renderer->setClipPlaneEquation(index, pe);
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_error(E_FAIL, e.what());
        return -1;
    }
}

MxClipPlane MxClipPlanes::item(const unsigned int &index) {
    return MxClipPlane(index);
}

MxClipPlane MxClipPlanes::create(const MxVector4f &pe) {
    MxClipPlane result(MxClipPlanes::len());
    MxSystem::getRenderer()->addClipPlaneEquation(pe);
    return result;
}

MxClipPlane MxClipPlanes::create(const MxVector3f &point, const MxVector3f &normal) {
    return MxClipPlanes::create(MxPlaneEquation(normal, point));
}

MxClipPlanes *getClipPlanes() {
    return &_clipPlanesObj;
}
