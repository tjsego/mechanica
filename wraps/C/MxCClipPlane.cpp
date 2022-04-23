/**
 * @file MxCClipPlane.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxClipPlane
 * @date 2022-04-11
 */

#include "MxCClipPlane.h"

#include "mechanica_c_private.h"

#include <rendering/MxClipPlane.hpp>


//////////////////
// Module casts //
//////////////////


namespace mx { 

MxClipPlane *castC(struct MxClipPlaneHandle *handle) {
    return castC<MxClipPlane, MxClipPlaneHandle>(handle);
}

}

#define MXCLIPPLANE_GET(handle, varname) \
    MxClipPlane *varname = mx::castC<MxClipPlane, MxClipPlaneHandle>(handle); \
    MXCPTRCHECK(varname);


/////////////////
// MxClipPlane //
/////////////////


HRESULT MxCClipPlane_getIndex(struct MxClipPlaneHandle *handle, int *index) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(index);
    *index = cp->index;
    return S_OK;
}

HRESULT MxCClipPlane_getPoint(struct MxClipPlaneHandle *handle, float **point) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(point);
    auto _point = cp->getPoint();
    MXVECTOR3_COPYFROM(_point, (*point));
    return S_OK;
}

HRESULT MxCClipPlane_getNormal(struct MxClipPlaneHandle *handle, float **normal) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(normal);
    auto _normal = cp->getNormal();
    MXVECTOR3_COPYFROM(_normal, (*normal));
    return S_OK;
}

HRESULT MxCClipPlane_getEquation(struct MxClipPlaneHandle *handle, float **pe) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(pe);
    auto _pe = cp->getEquation();
    MXVECTOR4_COPYFROM(_pe, (*pe));
    return S_OK;
}

HRESULT MxCClipPlane_setEquationE(struct MxClipPlaneHandle *handle, float *pe) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(pe);
    return cp->setEquation(MxVector4f::from(pe));
}

HRESULT MxCClipPlane_setEquationPN(struct MxClipPlaneHandle *handle, float *point, float *normal) {
    MXCLIPPLANE_GET(handle, cp);
    MXCPTRCHECK(point);
    MXCPTRCHECK(normal);
    return cp->setEquation(MxVector3f::from(point), MxVector3f::from(normal));
}

HRESULT MxCClipPlane_destroyCP(struct MxClipPlaneHandle *handle) {
    MXCLIPPLANE_GET(handle, cp);
    return cp->destroy();
}

HRESULT MxCClipPlane_destroy(struct MxClipPlaneHandle *handle) {
    return mx::capi::destroyHandle<MxClipPlane, MxClipPlaneHandle>(handle) ? S_OK : E_FAIL;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCClipPlanes_len(unsigned int *numCPs) {
    MXCPTRCHECK(numCPs);
    *numCPs = MxClipPlanes::len();
    return S_OK;
}

HRESULT MxCClipPlanes_item(struct MxClipPlaneHandle *handle, unsigned int index) {
    if(index >= MxClipPlanes::len()) 
        return E_FAIL;
    MXCPTRCHECK(handle);
    MxClipPlane *cp = new MxClipPlane(MxClipPlanes::item(index));
    handle->MxObj = (void*)cp;
    return S_OK;
}

HRESULT MxCClipPlanes_createE(struct MxClipPlaneHandle *handle, float *pe) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(pe);
    MxClipPlane *cp = new MxClipPlane(MxClipPlanes::create(MxVector4f::from(pe)));
    handle->MxObj = (void*)cp;
    return S_OK;
}

HRESULT MxCClipPlanes_createPN(struct MxClipPlaneHandle *handle, float *point, float *normal) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(point);
    MXCPTRCHECK(normal);
    MxClipPlane *cp = new MxClipPlane(MxClipPlanes::create(MxVector3f::from(point), MxVector3f::from(normal)));
    handle->MxObj = (void*)cp;
    return S_OK;
}

HRESULT MxCPlaneEquationFPN(float *point, float *normal, float **planeEq) {
    MXCPTRCHECK(point);
    MXCPTRCHECK(normal);
    MXCPTRCHECK(planeEq);
    MxVector4f _planeEq = MxPlaneEquation(MxVector3f::from(normal), MxVector3f::from(point));
    MXVECTOR4_COPYFROM(_planeEq, (*planeEq));
    return S_OK;
}

HRESULT MxCPlaneEquationTPN(float *planeEq, float **point, float **normal) {
    MXCPTRCHECK(planeEq);
    MXCPTRCHECK(point);
    MXCPTRCHECK(normal);
    MxVector3f _point, _normal;
    std::tie(_normal, _point) = MxPlaneEquation(MxVector4f::from(planeEq));
    MXVECTOR3_COPYFROM(_point, (*point));
    MXVECTOR3_COPYFROM(_normal, (*normal));
    return S_OK;
}
