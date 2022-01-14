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
    /** Index of the clip plane. Less than zero if clip plane has been destroyed. */
    int index;
    MxClipPlane(int i);

    /**
     * @brief Get the point of the clip plane
     * 
     * @return MxVector3f 
     */
    MxVector3f getPoint();

    /**
     * @brief Get the normal vector of the clip plane
     * 
     * @return MxVector3f 
     */
    MxVector3f getNormal();

    /**
     * @brief Get the coefficients of the plane equation of the clip plane
     * 
     * @return MxVector4f 
     */
    MxVector4f getEquation();

    /**
     * @brief Set the coefficients of the plane equation of the clip plane
     * 
     * @param pe 
     * @return HRESULT 
     */
    HRESULT setEquation(const MxVector4f &pe);

    /**
     * @brief Set the coefficients of the plane equation of the clip plane
     * using a point on the plane and its normal
     * 
     * @param point plane point
     * @param normal plane normal vector
     * @return HRESULT 
     */
    HRESULT setEquation(const MxVector3f &point, const MxVector3f &normal);

    /**
     * @brief Destroy the clip plane
     * 
     * @return HRESULT 
     */
    HRESULT destroy();
};

struct CAPI_EXPORT MxClipPlanes {

    /**
     * @brief Get the number of clip planes
     * 
     * @return int 
     */
    static int len();

    /**
     * @brief Get the coefficients of the equation of a clip plane
     * 
     * @param index index of the clip plane
     * @return const MxVector4f& 
     */
    static const MxVector4f &getClipPlaneEquation(const unsigned int &index);

    /**
     * @brief Set the coefficients of the equation of a clip plane. 
     * 
     * The clip plane must already exist
     * 
     * @param index index of the clip plane
     * @param pe coefficients of the plane equation of the clip plane
     * @return HRESULT 
     */
    static HRESULT setClipPlaneEquation(const unsigned int &index, const MxVector4f &pe);

    /**
     * @brief Get a clip plane by index
     * 
     * @param index index of the clip plane
     * @return MxClipPlane 
     */
    static MxClipPlane item(const unsigned int &index);

    /**
     * @brief Create a clip plane
     * 
     * @param pe coefficients of the equation of the plane
     * @return MxClipPlane 
     */
    static MxClipPlane create(const MxVector4f &pe);

    /**
     * @brief Create a clip plane
     * 
     * @param point point on the clip plane
     * @param normal normal of the clip plane
     * @return MxClipPlane 
     */
    static MxClipPlane create(const MxVector3f &point, const MxVector3f &normal);
};

/**
 * get a reference to the cut planes collection.
 */
MxClipPlanes *getClipPlanes();

#endif /* SRC_RENDERING_MXCUTPLANE_HPP_ */
