/*
 * MxPy.h
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */
// todo: decide whether to leave most of this here, or isolate

#pragma once
#ifndef SRC_MDCORE_SRC_MXPY_H_
#define SRC_MDCORE_SRC_MXPY_H_

#include <mx_port.h>
#include <../../types/mx_types.h>
#include <../../types/mx_cast.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Math/Matrix3.h>

#include <string>

std::string pyerror_str();
CAPI_FUNC(PyObject*) PyImport_ImportString(const std::string &name);
CAPI_FUNC(PyObject*) MxIPython_Get();
CAPI_FUNC(bool) Mx_TerminalInteractiveShell();
CAPI_FUNC(bool) Mx_ZMQInteractiveShell();

namespace mx {

    template<>
    Magnum::Vector2 cast(PyObject *obj);

    template<>
    Magnum::Vector3 cast(PyObject *obj);

    template<>
    Magnum::Vector4 cast(PyObject *obj);

    template<>
    Magnum::Vector2i cast(PyObject *obj);

    template<>
    Magnum::Vector3i cast(PyObject *obj);

    template<>
    MxVector2f cast(PyObject *obj);

    template<>
    MxVector3f cast(PyObject *obj);

    template<>
    MxVector4f cast(PyObject *obj);

    template<>
    MxVector2i cast(PyObject *obj);

    template<>
    MxVector3i cast(PyObject *obj);

    template<>
    PyObject* cast<int16_t, PyObject*>(const int16_t &i);

    template<>
    PyObject* cast<uint16_t, PyObject*>(const uint16_t &i);

    template<>
    PyObject* cast<uint32_t, PyObject*>(const uint32_t &i);

    template<>
    PyObject* cast<uint64_t, PyObject*>(const uint64_t &i);

    template<>
    PyObject* cast<float, PyObject*>(const float &f);

    template<>
    float cast(PyObject *obj);

    template<>
    PyObject* cast<bool, PyObject*>(const bool &f);

    template<>
    bool cast(PyObject *obj);

    template<>
    PyObject* cast<double, PyObject*>(const double &f);

    template<>
    double cast(PyObject *obj);

    template<>
    PyObject* cast<int, PyObject*>(const int &i);

    template<>
    int cast(PyObject *obj);

    template<>
    PyObject* cast<std::string, PyObject*>(const std::string &s);

    template<>
    std::string cast(PyObject *o);

    template<>
    int16_t cast(PyObject *o);

    template<>
    uint16_t cast(PyObject *o);

    template<>
    uint32_t cast(PyObject *o);

    template<>
    uint64_t cast(PyObject *o);

    /**
     * check if type can be converted
     */
    template <typename T>
    bool check(PyObject *o);

    // template <>
    // bool check<bool>(PyObject *o);

    // template <>
    // bool check<std::string>(PyObject *o);

    // template <>
    // bool check<float>(PyObject *o);


    /**
     * grab either the i'th arg from the args, or keywords.
     *
     * gets a reference to the object, NULL if not exist.
     */
    PyObject *py_arg(const char* name, int index, PyObject *_args, PyObject *_kwargs);

        /**
         * gets the __repr__ / __str__ representations of python objects
         */
    std::string repr(PyObject *o);
    std::string str(PyObject *o);

    /**
     * get the python error string, empty string if no error.
     */
    std::string pyerror_str();

    template<typename T>
    T arg(const char* name, int index, PyObject *args, PyObject *kwargs) {
        PyObject *value = py_arg(name, index, args, kwargs);
        if(value) {
            return cast<PyObject, T>(value);
        }
        throw std::runtime_error(std::string("missing argument ") + name);
    };

    template<typename T>
    T arg(const char* name, int index, PyObject *args, PyObject *kwargs, T deflt) {

        PyObject *value = py_arg(name, index, args, kwargs);
        if(value) {
            return cast<PyObject, T>(value);
        }
        return deflt;
    };

}

#endif /* SRC_MDCORE_SRC_MXPY_H_ */
