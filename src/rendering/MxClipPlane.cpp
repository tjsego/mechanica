/*
 * MxCutPlane.cpp
 *
 *  Created on: Mar 26, 2021
 *      Author: andy
 */

#include "MxClipPlane.hpp"
#include <Magnum/Math/Distance.h>
#include <MxSimulator.h>
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

int MxClipPlanes::len() {
    return MxSimulator::get()->getRenderer()->clipPlaneCount();
}

const MxVector4f &MxClipPlanes::getClipPlaneEquation(const unsigned int &index) {
    try {
        MxUniverseRenderer *renderer = MxSimulator::get()->getRenderer();
        
        if(index > renderer->clipPlaneCount()) mx_exp(std::range_error("index out of bounds"));
        return renderer->getClipPlaneEquation(index);
    }
    catch(const std::exception &e) {
        mx_error(E_FAIL, e.what());
        MxVector4f *result = new MxVector4f();
        return *result;
    }
}

HRESULT MxClipPlanes::setClipPlaneEquation(const unsigned int &index, const MxVector4f &pe) {
    try {
        MxUniverseRenderer *renderer = MxSimulator::get()->getRenderer();
        if(index > renderer->clipPlaneCount()) mx_exp(std::range_error("index out of bounds"));
        renderer->setClipPlaneEquation(index, pe);
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_error(E_FAIL, e.what());
        return -1;
    }
}

MxClipPlanes *getClipPlanes() {
    return &_clipPlanesObj;
}
