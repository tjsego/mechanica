/*
 * MxCutPlane.hpp
 *
 *  Created on: Mar 26, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_RENDERING_MXCUTPLANE_HPP_
#define SRC_RENDERING_MXCUTPLANE_HPP_

#include <MxParticle.h>
#include <space_cell.h>
#include "../types/mx_types.h"

#include <vector>

MxVector4f MxPlaneEquation(const MxVector3f &normal, const MxVector3f &point);
CAPI_FUNC(HRESULT) MxPlaneEquation(const MxVector3f &normal, const MxVector3f &point, float *result);

std::vector<MxVector4f> MxParsePlaneEquation(const std::vector<std::tuple<MxVector3f, MxVector3f> > &clipPlanes);
CAPI_FUNC(HRESULT) MxParsePlaneEquation(const std::vector<std::tuple<MxVector3f, MxVector3f> > &clipPlanes, MxVector4f *result);

std::tuple<MxVector3f, MxVector3f> MxPlaneEquation(const MxVector4f &planeEq);

struct MxClipPlane
{
    int index;
    MxClipPlane(int i);
};

struct CAPI_EXPORT MxClipPlanes {
    static int len();

    static const MxVector4f &getClipPlaneEquation(const unsigned int &index);
    static HRESULT setClipPlaneEquation(const unsigned int &index, const MxVector4f &pe);
};

/**
 * get a reference to the cut planes collection.
 */
MxClipPlanes *getClipPlanes();

#endif /* SRC_RENDERING_MXCUTPLANE_HPP_ */
