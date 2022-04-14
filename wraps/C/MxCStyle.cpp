/**
 * @file MxCStyle.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxStyle
 * @date 2022-03-30
 */

#include "MxCStyle.h"

#include "mechanica_c_private.h"

#include <rendering/MxStyle.hpp>
#include <MxParticle.h>

namespace mx { 

MxStyle *castC(struct MxStyleHandle *handle) {
    return castC<MxStyle, MxStyleHandle>(handle);
}

}

#define MXSTYLE_GET(handle) \
    MxStyle *style = mx::castC<MxStyle, MxStyleHandle>(handle); \
    MXCPTRCHECK(style);


/////////////
// MxStyle //
/////////////


HRESULT MxCStyle_init(struct MxStyleHandle *handle) {
    MXCPTRCHECK(handle);
    MxStyle *style = new MxStyle();
    handle->MxObj = (void*)style;
    return S_OK;
}

HRESULT MxCStyle_initC(struct MxStyleHandle *handle, float *color, bool visible) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(color);
    Magnum::Color3 _color = Magnum::Color3::from(color);
    MxStyle *style = new MxStyle(&_color, visible);
    handle->MxObj = (void*)style;
    return S_OK;
}

HRESULT MxCStyle_initS(struct MxStyleHandle *handle, const char *colorName, bool visible) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(colorName);
    MxStyle *style = new MxStyle(colorName, visible);
    handle->MxObj = (void*)style;
    return S_OK;
}

HRESULT MxCStyle_destroy(struct MxStyleHandle *handle) {
    return mx::capi::destroyHandle<MxStyle, MxStyleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT MxCStyle_setColor(struct MxStyleHandle *handle, const char *colorName) {
    MXSTYLE_GET(handle);
    MXCPTRCHECK(colorName);
    return style->setColor(colorName);
}

HRESULT MxCStyle_getVisible(struct MxStyleHandle *handle, bool *visible) {
    MXSTYLE_GET(handle);
    MXCPTRCHECK(visible);
    *visible = style->getVisible();
    return S_OK;
}

HRESULT MxCStyle_setVisible(struct MxStyleHandle *handle, bool visible) {
    MXSTYLE_GET(handle);
    style->setVisible(visible);
    return S_OK;
}

HRESULT MxCStyle_newColorMapper(struct MxStyleHandle *handle, 
                                struct MxParticleTypeHandle *partType,
                                const char *speciesName, 
                                const char *name, 
                                float min, 
                                float max) 
{
    MXSTYLE_GET(handle);
    MXCPTRCHECK(partType); MXCPTRCHECK(partType->MxObj);
    MXCPTRCHECK(speciesName);
    MXCPTRCHECK(name);
    style->newColorMapper((MxParticleType*)partType->MxObj, speciesName, name, min, max);
    return S_OK;
}

HRESULT MxCStyle_toString(struct MxStyleHandle *handle, char **str, unsigned int *numChars) {
    MXSTYLE_GET(handle);
    return mx::capi::str2Char(style->toString(), str, numChars);
}

HRESULT MxCStyle_fromString(struct MxStyleHandle *handle, const char *str) {
    MXCPTRCHECK(handle);
    MXCPTRCHECK(str);
    MxStyle *style = MxStyle::fromString(str);
    MXCPTRCHECK(style);
    handle->MxObj = (void*)style;
    return S_OK;
}
