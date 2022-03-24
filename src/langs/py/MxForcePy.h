/**
 * @file MxForcePy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxForce
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXFORCEPY_H_
#define _SRC_LANGS_PY_MXFORCEPY_H_


#include "MxPy.h"
#include <MxForce.h>


struct MxConstantForcePy : MxConstantForce {
    PyObject *callable;

    MxConstantForcePy();
    MxConstantForcePy(const MxVector3f &f, const float &period=std::numeric_limits<float>::max());

    /**
     * @brief Creates an instance from an underlying custom python function
     * 
     * @param f python function. Takes no arguments and returns a three-component vector. 
     * @param period period at which the force is updated. 
     */
    MxConstantForcePy(PyObject *f, const float &period=std::numeric_limits<float>::max());
    virtual ~MxConstantForcePy();

    void onTime(double time);
    MxVector3f getValue();

    void setValue(PyObject *_userFunc=NULL);

    /**
     * @brief Convert basic force to MxConstantForcePy. 
     * 
     * If the basic force is not a MxConstantForcePy, then NULL is returned. 
     * 
     * @param f 
     * @return MxConstantForcePy* 
     */
    static MxConstantForcePy *fromForce(MxForce *f);

};

namespace mx { namespace io { 

template <>
HRESULT toFile(const MxConstantForcePy &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxConstantForcePy *dataElement);

}}

#endif // _SRC_LANGS_PY_MXFORCEPY_H_
