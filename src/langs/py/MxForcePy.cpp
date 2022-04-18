/**
 * @file MxForcePy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxForce
 * @date 2022-03-23
 * 
 */

#include "MxForcePy.h"

#include <MxLogger.h>


MxVector3f pyConstantForceFunction(PyObject *callable) {
    Log(LOG_TRACE);

    PyObject *result = PyObject_CallObject(callable, NULL);

    if(result == NULL) {
        PyObject *err = PyErr_Occurred();
        Log(LOG_CRITICAL) << pyerror_str();
        PyErr_Clear();
        return MxVector3f();
    }
    MxVector3f out = mx::cast<PyObject, MxVector3f>(result);
    Py_DECREF(result);
    return out;
}

MxConstantForcePy::MxConstantForcePy() : 
    MxConstantForce() 
{
    type = FORCE_CONSTANT;
}

MxConstantForcePy::MxConstantForcePy(const MxVector3f &f, const float &period) : 
    MxConstantForce(f, period)
{
    type = FORCE_CONSTANT;
    callable = NULL;
}

MxConstantForcePy::MxConstantForcePy(PyObject *f, const float &period) : 
    MxConstantForce(), 
    callable(f)
{
    type = FORCE_CONSTANT;

    setPeriod(period);
    if(PyList_Check(f)) {
        MxVector3f fv = mx::cast<PyObject, MxVector3f>(f);
        callable = NULL;
        MxConstantForce::setValue(fv);
    }
    else if(callable) {
        Py_IncRef(callable);
    }
}

MxConstantForcePy::~MxConstantForcePy(){
    if(callable) Py_DecRef(callable);
}

void MxConstantForcePy::onTime(double time)
{
    if(callable && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue(callable);
    }
}

MxVector3f MxConstantForcePy::getValue() {
    if(callable && callable != Py_None) return pyConstantForceFunction(callable);
    return force;
}

void MxConstantForcePy::setValue(PyObject *_userFunc) {
    if(_userFunc) callable = _userFunc;
    if(callable && callable != Py_None) MxConstantForce::setValue(getValue());
}

MxConstantForcePy *MxConstantForcePy::fromForce(MxForce *f) {
    if(f->type != FORCE_CONSTANT) 
        return 0;
    return (MxConstantForcePy*)f;
}


namespace mx { namespace io {

#define MXFORCEPYIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXFORCEPYIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxConstantForcePy &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEPYIOTOEASY(fe, "type", dataElement.type);
    MXFORCEPYIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEPYIOTOEASY(fe, "updateInterval", dataElement.updateInterval);
    MXFORCEPYIOTOEASY(fe, "lastUpdate", dataElement.lastUpdate);
    MXFORCEPYIOTOEASY(fe, "force", dataElement.force);

    fileElement->type = "ConstantPyForce";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxConstantForcePy *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEPYIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEPYIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEPYIOFROMEASY(feItr, fileElement.children, metaData, "updateInterval", &dataElement->updateInterval);
    MXFORCEPYIOFROMEASY(feItr, fileElement.children, metaData, "lastUpdate", &dataElement->lastUpdate);
    MXFORCEPYIOFROMEASY(feItr, fileElement.children, metaData, "force", &dataElement->force);
    dataElement->userFunc = NULL;
    dataElement->callable = NULL;

    return S_OK;
}


}}
