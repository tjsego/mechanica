/**
 * @file MxCSystem.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSystem
 * @date 2022-04-04
 */

#include "MxCSystem.h"

#include "mechanica_c_private.h"

#include <MxSystem.h>


HRESULT MxCSystem_imageData(char **imgData, size_t *imgSize) {
    MXCPTRCHECK(imgData);
    MXCPTRCHECK(imgSize);
    char *_imgData;
    std::tie(_imgData, *imgSize) = MxSystem::imageData();
    *imgData = _imgData;
    return S_OK;
}

HRESULT MxCSystem_screenshot(const char *filePath) {
    return MxCSystem_screenshot(filePath);
}

HRESULT MxCSystem_screenshotS(const char *filePath, bool decorate, float *bgcolor) {
    if(!bgcolor) 
        return E_FAIL;
    return MxSystem::screenshot(filePath, decorate, MxVector3f::from(bgcolor));
}

HRESULT MxCSystem_contextHasCurrent(bool *current) {
    MXCPTRCHECK(current);
    *current = MxSystem::contextHasCurrent();
    return S_OK;
}

HRESULT MxCSystem_contextMakeCurrent() {
    return MxSystem::contextMakeCurrent();
}

HRESULT MxCSystem_contextRelease() {
    return MxSystem::contextRelease();
}

HRESULT MxCSystem_cameraMoveTo(float *eye, float *center, float *up) {
    if(!eye || !center || !up) 
        return E_FAIL;
    return MxSystem::cameraMoveTo(MxVector3f::from(eye), MxVector3f::from(center), MxVector3f::from(up));
}

HRESULT MxCSystem_cameraMoveToR(float *center, float *rotation, float zoom) {
    if(!center || !rotation) 
        return E_FAIL;
    return MxSystem::cameraMoveTo(MxVector3f::from(center), MxQuaternionf{MxVector3f(rotation[0], rotation[1], rotation[2]), rotation[3]}, zoom);
}

HRESULT MxCSystem_cameraViewBottom() {
    return MxSystem::cameraViewBottom();
}

HRESULT MxCSystem_cameraViewTop() {
    return MxSystem::cameraViewTop();
}

HRESULT MxCSystem_cameraViewLeft() {
    return MxSystem::cameraViewLeft();
}

HRESULT MxCSystem_cameraViewRight() {
    return MxSystem::cameraViewRight();
}

HRESULT MxCSystem_cameraViewBack() {
    return MxSystem::cameraViewBack();
}

HRESULT MxCSystem_cameraViewFront() {
    return MxSystem::cameraViewFront();
}

HRESULT MxCSystem_cameraReset() {
    return MxSystem::cameraReset();
}

HRESULT MxCSystem_cameraRotateMouse(int x, int y) {
    return MxSystem::cameraRotateMouse({x, y});
}

HRESULT MxCSystem_cameraTranslateMouse(int x, int y) {
    return MxSystem::cameraTranslateMouse({x, y});
}

HRESULT MxCSystem_cameraTranslateDown() {
    return MxSystem::cameraTranslateDown();
}

HRESULT MxCSystem_cameraTranslateUp() {
    return MxSystem::cameraTranslateUp();
}

HRESULT MxCSystem_cameraTranslateRight() {
    return MxSystem::cameraTranslateRight();
}

HRESULT MxCSystem_cameraTranslateLeft() {
    return MxSystem::cameraTranslateLeft();
}

HRESULT MxCSystem_cameraTranslateForward() {
    return MxSystem::cameraTranslateForward();
}

HRESULT MxCSystem_cameraTranslateBackward() {
    return MxSystem::cameraTranslateBackward();
}

HRESULT MxCSystem_cameraRotateDown() {
    return MxSystem::cameraRotateDown();
}

HRESULT MxCSystem_cameraRotateUp() {
    return MxSystem::cameraRotateUp();
}

HRESULT MxCSystem_cameraRotateLeft() {
    return MxSystem::cameraRotateLeft();
}

HRESULT MxCSystem_cameraRotateRight() {
    return MxSystem::cameraRotateRight();
}

HRESULT MxCSystem_cameraRollLeft() {
    return MxSystem::cameraRollLeft();
}

HRESULT MxCSystem_cameraRollRight() {
    return MxSystem::cameraRollRight();
}

HRESULT MxCSystem_cameraZoomIn() {
    return MxSystem::cameraZoomIn();
}

HRESULT MxCSystem_cameraZoomOut() {
    return MxSystem::cameraZoomOut();
}

HRESULT MxCSystem_cameraInitMouse(int x, int y) {
    return MxSystem::cameraInitMouse({x, y});
}

HRESULT MxCSystem_cameraTranslateBy(float x, float y) {
    return MxSystem::cameraTranslateBy({x, y});
}

HRESULT MxCSystem_cameraZoomBy(float delta) {
    return MxSystem::cameraZoomBy(delta);
}

HRESULT MxCSystem_cameraZoomTo(float distance) {
    return MxSystem::cameraZoomTo(distance);
}

HRESULT MxCSystem_cameraRotateToAxis(float *axis, float distance) {
    if(!axis) 
        return E_FAIL;
    return MxSystem::cameraRotateToAxis(MxVector3f::from(axis), distance);
}

HRESULT MxCSystem_cameraRotateToEulerAngle(float *angles) {
    if(!angles) 
        return E_FAIL;
    return MxSystem::cameraRotateToEulerAngle(MxVector3f::from(angles));
}

HRESULT MxCSystem_cameraRotateByEulerAngle(float *angles) {
    if(!angles) 
        return E_FAIL;
    return MxSystem::cameraRotateByEulerAngle(MxVector3f::from(angles));
}

HRESULT MxCSystem_cameraCenter(float **center) {
    if(!center) 
        return E_FAIL;
    auto _center = MxSystem::cameraCenter();
    MXVECTOR3_COPYFROM(_center, (*center));
    return S_OK;
}

HRESULT MxCSystem_cameraRotation(float **rotation) {
    if(!rotation) 
        return E_FAIL;
    auto _rotation = MxSystem::cameraRotation();
    auto axis = _rotation.axis();
    MXVECTOR3_COPYFROM(axis, (*rotation));
    *rotation[3] = _rotation.angle();
    return S_OK;
}

HRESULT MxCSystem_cameraZoom(float *zoom) {
    MXCPTRCHECK(zoom);
    *zoom = MxSystem::cameraZoom();
    return S_OK;
}

HRESULT MxCSystem_getAmbientColor(float **color) {
    MXCPTRCHECK(color);
    auto _color = MxSystem::getAmbientColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setAmbientColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setAmbientColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getDiffuseColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getDiffuseColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setDiffuseColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setDiffuseColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getSpecularColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getSpecularColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setSpecularColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setSpecularColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getShininess(float *shininess) {
    MXCPTRCHECK(shininess);
    *shininess = MxSystem::getShininess();
    return S_OK;
}

HRESULT MxCSystem_setShininess(float shininess) {
    return MxSystem::setShininess(shininess);
}

HRESULT MxCSystem_getGridColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getGridColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setGridColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setGridColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getSceneBoxColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getSceneBoxColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setSceneBoxColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setSceneBoxColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getLightDirection(float **lightDir) {
    if(!lightDir) 
        return E_FAIL;
    auto _lightDir = MxSystem::getLightDirection();
    MXVECTOR3_COPYFROM(_lightDir, (*lightDir));
    return S_OK;
}

HRESULT MxCSystem_setLightDirection(float *lightDir) {
    if(!lightDir) 
        return E_FAIL;
    return MxSystem::setLightDirection(MxVector3f::from(lightDir));
}

HRESULT MxCSystem_getLightColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getLightColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setLightColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setLightColor(MxVector3f::from(color));
}

HRESULT MxCSystem_getBackgroundColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getBackgroundColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setBackgroundColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setBackgroundColor(MxVector3f::from(color));
}

HRESULT MxCSystem_decorated(bool *decorated) {
    MXCPTRCHECK(decorated);
    *decorated = MxSystem::decorated();
    return S_OK;
}

HRESULT MxCSystem_decorateScene(bool decorate) {
    return MxSystem::decorateScene(decorate);
}

HRESULT MxCSystem_showingDiscretization(bool *showing) {
    MXCPTRCHECK(showing);
    *showing = MxSystem::showingDiscretization();
    return S_OK;
}

HRESULT MxCSystem_showDiscretization(bool show) {
    return MxSystem::showDiscretization(show);
}

HRESULT MxCSystem_getDiscretizationColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = MxSystem::getDiscretizationColor();
    MXVECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT MxCSystem_setDiscretizationColor(float *color) {
    if(!color) 
        return E_FAIL;
    return MxSystem::setDiscretizationColor(MxVector3f::from(color));
}

HRESULT MxCSystem_viewReshape(int sizex, int sizey) {
    return MxSystem::viewReshape({sizex, sizey});
}

HRESULT MxCSystem_getCPUInfo(char ***names, bool **flags, unsigned int *numNames) {
    MXCPTRCHECK(names);
    MXCPTRCHECK(flags);
    MXCPTRCHECK(numNames);
    auto cpu_info = MxSystem::cpu_info();
    *numNames = cpu_info.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        bool *_flags = (bool*)malloc(*numNames * sizeof(bool));
        if(!_names || !_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : cpu_info) {
            char *_c = new char[itr.first.size() + 1];
            std::strcpy(_c, itr.first.c_str());
            _names[i] = _c;
            _flags[i] = itr.second;
            i++;
        }
        *names = _names;
        *flags = _flags;
    }
    return S_OK;
}

HRESULT MxCSystem_getCompileFlags(char ***flags, unsigned int *numFlags) {
    MXCPTRCHECK(flags);
    MXCPTRCHECK(numFlags);
    auto compile_flags = MxSystem::compile_flags();
    *numFlags = compile_flags.size();
    if(*numFlags > 0) {
        char **_flags = (char**)malloc(*numFlags * sizeof(char*));
        if(!_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(std::string &_s : compile_flags) {
            char *_c = new char[_s.size() + 1];
            std::strcpy(_c, _s.c_str());
            _flags[i] = _c;
            i++;
        }
        *flags = _flags;
    }
    return S_OK;
}

HRESULT MxCSystem_getGLInfo(char ***names, char ***values, unsigned int *numNames) {
    MXCPTRCHECK(names);
    MXCPTRCHECK(values);
    MXCPTRCHECK(numNames);
    auto gl_info = MxSystem::gl_info();
    *numNames = gl_info.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        char **_values = (char**)malloc(*numNames * sizeof(char*));
        if(!_names || !_values) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : gl_info) {
            char *_sn = new char[itr.first.size() + 1];
            char *_sv = new char[itr.second.size() + 1];
            std::strcpy(_sn, itr.first.c_str());
            std::strcpy(_sv, itr.second.c_str());
            i++;
        }
        *names = _names;
        *values = _values;
    }
    return S_OK;
}

HRESULT MxCSystem_getEGLInfo(char **info, unsigned int *numChars) {
    MXCPTRCHECK(info);
    MXCPTRCHECK(numChars);
    return mx::capi::str2Char(MxSystem::egl_info(), info, numChars);
}
