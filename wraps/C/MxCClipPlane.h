/**
 * @file MxCClipPlane.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxClipPlane
 * @date 2022-04-11
 */

#ifndef _WRAPS_C_MXCCLIPPLANE_H_
#define _WRAPS_C_MXCCLIPPLANE_H_

#include <mx_port.h>

// Handles

/**
 * @brief Handle to a @ref MxClipPlane instance
 * 
 */
struct CAPI_EXPORT MxClipPlaneHandle {
    void *MxObj;
};


/////////////////
// MxClipPlane //
/////////////////


/**
 * @brief Get the index of the clip plane.
 * 
 * @param handle populated handle
 * @param index index of the clip plane; Less than zero if clip plane has been destroyed.
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_getIndex(struct MxClipPlaneHandle *handle, int *index);

/**
 * @brief Get the point of the clip plane
 * 
 * @param handle populated handle
 * @param point point of the clip plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClipPlane_getPoint(struct MxClipPlaneHandle *handle, float **point);

/**
 * @brief Get the normal vector of the clip plane
 * 
 * @param handle populated handle
 * @param normal normal of the clip plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClipPlane_getNormal(struct MxClipPlaneHandle *handle, float **normal);

/**
 * @brief Get the coefficients of the plane equation of the clip plane
 * 
 * @param handle populated handle
 * @param pe plane equation coefficients
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_getEquation(struct MxClipPlaneHandle *handle, float **pe);

/**
 * @brief Set the coefficients of the plane equation of the clip plane
 * 
 * @param handle populated handle
 * @param pe plane equation coefficients
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_setEquationE(struct MxClipPlaneHandle *handle, float *pe);

/**
 * @brief Set the coefficients of the plane equation of the clip plane
 * using a point on the plane and its normal
 * 
 * @param handle populated handle
 * @param point plane point
 * @param normal plane normal vector
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_setEquationPN(struct MxClipPlaneHandle *handle, float *point, float *normal);

/**
 * @brief Destroy the clip plane
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_destroyCP(struct MxClipPlaneHandle *handle);

/**
 * @brief Destroy the handle instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlane_destroy(struct MxClipPlaneHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the number of clip planes
 * 
 * @param numCPs number of clip planes
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlanes_len(unsigned int *numCPs);

/**
 * @brief Get a clip plane by index
 * 
 * @param handle handle to populate
 * @param index index of the clip plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlanes_item(struct MxClipPlaneHandle *handle, unsigned int index);

/**
 * @brief Create a clip plane from the coefficients of the equation of the plane
 * 
 * @param handle handle to populate
 * @param pe coefficients of the equation of the plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlanes_createE(struct MxClipPlaneHandle *handle, float *pe);

/**
 * @brief Create a clip plane from a point and normal of the plane
 * 
 * @param handle handle to populate
 * @param point point on the clip plane
 * @param normal normal of the clip plane
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClipPlanes_createPN(struct MxClipPlaneHandle *handle, float *point, float *normal);

/**
 * @brief Calculate the coefficients of a plane equation from a point and normal of the plane
 * 
 * @param point point on the plane
 * @param normal normal of the plane
 * @param planeEq plane equation coefficients
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPlaneEquationFPN(float *point, float *normal, float **planeEq);

/**
 * @brief Calculate a point and normal of a plane equation
 * 
 * @param planeEq coefficients of the plane equation
 * @param point point on the plane
 * @param normal normal of the plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCPlaneEquationTPN(float *planeEq, float **point, float **normal);

#endif // _WRAPS_C_MXCCLIPPLANE_H_