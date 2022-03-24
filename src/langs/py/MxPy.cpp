/*
 * MxPy.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: andy
 */

#include "MxPy.h"

#include <MxLogger.h>


bool hasPython() {
    if(!Py_IsInitialized()) {
        Log(LOG_DEBUG) << "Python not initialized";
        return false;
    }
    return true;
}

#define MxCHECKPYRET(x) if(!hasPython()) return x;

PyObject *PyImport_ImportString(const std::string &name) {
    MxCHECKPYRET(0)

    PyObject *s = mx::cast<std::string, PyObject*>(name);
    PyObject *mod = PyImport_Import(s);
    Py_DECREF(s);
    return mod;
}

std::string pyerror_str()
{
    MxCHECKPYRET("")

    std::string result;
    // get the error details
    PyObject *pExcType = NULL , *pExcValue = NULL , *pExcTraceback = NULL ;
    PyErr_Fetch( &pExcType , &pExcValue , &pExcTraceback ) ;
    if ( pExcType != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcType ) ;
        
        PyObject * str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC type: ") + PyBytes_AsString(str);
        Py_DECREF(str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcType ) ;
    }
    if ( pExcValue != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcValue ) ;

        PyObject * str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC value: ") + PyBytes_AsString(str);
        Py_DECREF(str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcValue ) ;
    }
    if ( pExcTraceback != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcValue ) ;
        
        PyObject * str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC traceback: ") + PyBytes_AsString(str);
        Py_DECREF(str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcTraceback ) ;
    }
    
    return result;
}

PyObject* MxIPython_Get() {
    MxCHECKPYRET(0)

    PyObject* moduleString = PyUnicode_FromString("IPython.core.getipython");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        PyObject *err = PyErr_Occurred();
        
        Log(LOG_DEBUG) << "could not import IPython.core.getipython"
            << ", "
            << pyerror_str()
            << ", returning NULL";
        PyErr_Clear();
        Py_DECREF(moduleString);
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* get_ipython = PyObject_GetAttrString(module,(char*)"get_ipython");
    
    if(!get_ipython) {
        PyObject *err = PyErr_Occurred();
        Log(LOG_WARNING) << "PyObject_GetAttrString(\"get_ipython\") failed: "
            << pyerror_str()
            << ", returning NULL";
        PyErr_Clear();
        Py_DECREF(moduleString);
        Py_DECREF(module);
        return NULL;
    }

    PyObject* result = PyObject_CallObject(get_ipython, NULL);
    
    if(result == NULL) {
        PyObject* err = PyErr_Occurred();
        std::string str = "error calling IPython.core.getipython.get_ipython(): ";
        str += pyerror_str();
        Log(LOG_FATAL) << str;
        PyErr_Clear();
    }
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(get_ipython);
    
    Log(LOG_TRACE);
    return result;
}

bool MxPy_TerminalInteractiveShell() {
    MxCHECKPYRET(false)

    PyObject* ipy = MxIPython_Get();
    bool result = false;

    if (ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {
        result = true;
    }
    
    Log(LOG_TRACE) << "returning: " << result;
    Py_XDECREF(ipy);
    return result;
}

bool MxPy_ZMQInteractiveShell() {
    PyObject* ipy = MxIPython_Get();
    bool result = false;

    if (ipy && strcmp("ZMQInteractiveShell", ipy->ob_type->tp_name) == 0) {
        result = true;
    }
    
    Log(LOG_TRACE) << "returning: " << result;
    Py_XDECREF(ipy);
    return result;
}

namespace mx {
    
    std::string pyerror_str()
    {
        MxCHECKPYRET("")

        std::string result;
        // get the error details
        PyObject *pExcType = NULL , *pExcValue = NULL , *pExcTraceback = NULL ;
        PyErr_Fetch( &pExcType , &pExcValue , &pExcTraceback ) ;
        if ( pExcType != NULL )
        {
            PyObject* pRepr = PyObject_Repr( pExcType ) ;
            
            PyObject * str=PyUnicode_AsASCIIString(pRepr);
            result += std::string("EXC type: ") + PyBytes_AsString(str);
            Py_DECREF(str);
            
            Py_DecRef( pRepr ) ;
            Py_DecRef( pExcType ) ;
        }
        if ( pExcValue != NULL )
        {
            PyObject* pRepr = PyObject_Repr( pExcValue ) ;

            PyObject * str=PyUnicode_AsASCIIString(pRepr);
            result += std::string("EXC value: ") + PyBytes_AsString(str);
            Py_DECREF(str);
            
            Py_DecRef( pRepr ) ;
            Py_DecRef( pExcValue ) ;
        }
        if ( pExcTraceback != NULL )
        {
            PyObject* pRepr = PyObject_Repr( pExcValue ) ;
            
            PyObject * str=PyUnicode_AsASCIIString(pRepr);
            result += std::string("EXC traceback: ") + PyBytes_AsString(str);
            Py_DECREF(str);
            
            Py_DecRef( pRepr ) ;
            Py_DecRef( pExcTraceback ) ;
        }
        
        return result;
    }

    template<>
    float cast(PyObject *obj) {
        MxCHECKPYRET(0)

        if(PyNumber_Check(obj)) {
            return PyFloat_AsDouble(obj);
        }
        throw std::domain_error("can not convert to number");
    }

    static Magnum::Vector3 vector3_from_list(PyObject *obj) {
        Magnum::Vector3 result = {};
        
        MxCHECKPYRET(result)

        if(PyList_Size(obj) != 3) {
            throw std::domain_error("error, must be length 3 list to convert to vector3");
        }
        
        for(int i = 0; i < 3; ++i) {
            PyObject *item = PyList_GetItem(obj, i);
            if(PyNumber_Check(item)) {
                result[i] = PyFloat_AsDouble(item);
            }
            else {
                throw std::domain_error("error, can not convert list item to number");
            }
        }
        
        return result;
    }

    static Magnum::Vector4 vector4_from_list(PyObject *obj) {
        Magnum::Vector4 result = {};

        MxCHECKPYRET(result)
        
        if(PyList_Size(obj) != 4) {
            throw std::domain_error("error, must be length 3 list to convert to vector3");
        }
        
        for(int i = 0; i < 4; ++i) {
            PyObject *item = PyList_GetItem(obj, i);
            if(PyNumber_Check(item)) {
                result[i] = PyFloat_AsDouble(item);
            }
            else {
                throw std::domain_error("error, can not convert list item to number");
            }
        }
        
        return result;
    }

    static Magnum::Vector2 vector2_from_list(PyObject *obj) {
        Magnum::Vector2 result = {};
        
        MxCHECKPYRET(result)

        if(PyList_Size(obj) != 2) {
            throw std::domain_error("error, must be length 2 list to convert to vector3");
        }
        
        for(int i = 0; i < 2; ++i) {
            PyObject *item = PyList_GetItem(obj, i);
            if(PyNumber_Check(item)) {
                result[i] = PyFloat_AsDouble(item);
            }
            else {
                throw std::domain_error("error, can not convert list item to number");
            }
        }
        
        return result;
    }

    static Magnum::Vector3i vector3i_from_list(PyObject *obj) {
        Magnum::Vector3i result = {};

        MxCHECKPYRET(result)

        if(PyList_Size(obj) != 3) {
            throw std::domain_error("error, must be length 3 list to convert to vector3");
        }
        
        for(int i = 0; i < 3; ++i) {
            PyObject *item = PyList_GetItem(obj, i);
            if(PyNumber_Check(item)) {
                result[i] = PyLong_AsLong(item);
            }
            else {
                throw std::domain_error("error, can not convert list item to number");
            }
        }
        
        return result;
    }
        
    static Magnum::Vector2i vector2i_from_list(PyObject *obj) {
        Magnum::Vector2i result = {};

        MxCHECKPYRET(result)

        if(PyList_Size(obj) != 2) {
            throw std::domain_error("error, must be length 2 list to convert to vector2");
        }
        
        for(int i = 0; i < 2; ++i) {
            PyObject *item = PyList_GetItem(obj, i);
            if(PyNumber_Check(item)) {
                result[i] = PyLong_AsLong(item);
            }
            else {
                throw std::domain_error("error, can not convert list item to number");
            }
        }
        
        return result;
    }

    template<>
    Magnum::Vector3 cast(PyObject *obj) {

        MxCHECKPYRET(Magnum::Vector3(0))

        if(PyList_Check(obj)) {
            return vector3_from_list(obj);
        }
        throw std::domain_error("can not convert non-list to vector");
    }

    template<>
    Magnum::Vector4 cast(PyObject *obj) {

        MxCHECKPYRET(Magnum::Vector4(0))

        if(PyList_Check(obj)) {
            return vector4_from_list(obj);
        }
        throw std::domain_error("can not convert non-list to vector");
    }

    template<>
    Magnum::Vector2 cast(PyObject *obj) {

        MxCHECKPYRET(Magnum::Vector2(0))

        if(PyList_Check(obj)) {
            return vector2_from_list(obj);
        }
        throw std::domain_error("can not convert non-list to vector");
    }
        
    template<>
    Magnum::Vector3i cast(PyObject *obj) {

        MxCHECKPYRET(Magnum::Vector3i(0))

        if(PyList_Check(obj)) {
            return vector3i_from_list(obj);
        }
        throw std::domain_error("can not convert non-list to vector");
    }
        
    template<>
    Magnum::Vector2i cast(PyObject *obj) {

        MxCHECKPYRET(Magnum::Vector2i(0))

        if(PyList_Check(obj)) {
            return vector2i_from_list(obj);
        }
        throw std::domain_error("can not convert non-list to vector");
    }

    template<>
    MxVector2f cast(PyObject *obj) { return MxVector2f(cast<PyObject, Magnum::Vector2>(obj)); }

    template<>
    MxVector3f cast(PyObject *obj) { return MxVector3f(cast<PyObject, Magnum::Vector3>(obj)); }

    template<>
    MxVector4f cast(PyObject *obj) { return MxVector4f(cast<PyObject, Magnum::Vector4>(obj)); }

    template<>
    MxVector2i cast(PyObject *obj) { return MxVector2i(cast<PyObject, Magnum::Vector2i>(obj)); }

    template<>
    MxVector3i cast(PyObject *obj) { return MxVector3i(cast<PyObject, Magnum::Vector3i>(obj)); }

    template<>
    PyObject* cast<float, PyObject*>(const float &f) {
        MxCHECKPYRET(0)

        return PyFloat_FromDouble(f);
    }

    template<>
    PyObject* cast<int16_t, PyObject*>(const int16_t &i) {
        MxCHECKPYRET(0)

        return PyLong_FromLong(i);
    }

    template<>
    PyObject* cast<uint16_t, PyObject*>(const uint16_t &i) {
        MxCHECKPYRET(0)

        return PyLong_FromLong(i);
    }

    template<>
    PyObject* cast<uint32_t, PyObject*>(const uint32_t &i) {
        MxCHECKPYRET(0)

        return PyLong_FromLong(i);
    }

    template<>
    PyObject* cast<uint64_t, PyObject*>(const uint64_t &i) {
        MxCHECKPYRET(0)

        return PyLong_FromLong(i);
    }

    template<>
    bool cast(PyObject *obj) {
        MxCHECKPYRET(0)

        if(PyBool_Check(obj)) {
            return obj == Py_True ? true : false;
        }
        throw std::domain_error("can not convert to boolean");
    }

    template<>
    PyObject* cast<bool, PyObject*>(const bool &b) {
        MxCHECKPYRET(0)

        if(b) {
            Py_RETURN_TRUE;
        }
        else {
            Py_RETURN_FALSE;
        }
    }

    template <>
    bool check<bool>(PyObject *o) {
        MxCHECKPYRET(0)

        return PyBool_Check(o);
    }

    PyObject *py_arg(const char* name, int index, PyObject *_args, PyObject *_kwargs) {
        MxCHECKPYRET(0)

        PyObject *kwobj = _kwargs ?  PyDict_GetItemString(_kwargs, name) : NULL;
        PyObject *aobj = _args && (PyTuple_Size(_args) > index) ? PyTuple_GetItem(_args, index) : NULL;
        
        if(aobj && kwobj) {
            std::string msg = std::string("Error, argument \"") + name + "\" given both as a keyword and positional";
            throw std::logic_error(msg.c_str());
        }
        
        return aobj ? aobj : kwobj;
    }
    
    template<>
    PyObject* cast<double, PyObject*>(const double &f){
        MxCHECKPYRET(0)

        return PyFloat_FromDouble(f);
    }

    template<>
    double cast(PyObject *obj) {
        MxCHECKPYRET(0)

        if(PyNumber_Check(obj)) {
            return PyFloat_AsDouble(obj);
        }
        throw std::domain_error("can not convert to number");
    }

    template<>
    PyObject* cast<int, PyObject*>(const int &i) {
        MxCHECKPYRET(0)

        return PyLong_FromLong(i);
    }

    template<>
    int cast(PyObject *obj){
        MxCHECKPYRET(0)

        if(PyNumber_Check(obj)) {
            return PyLong_AsLong(obj);
        }
        throw std::domain_error("can not convert to number");
    }

    template<>
    PyObject* cast<std::string, PyObject*>(const std::string &s) {
        MxCHECKPYRET(0)

        return PyUnicode_FromString(s.c_str());
    }

    template<>
    std::string cast(PyObject *o) {
        MxCHECKPYRET("")

        if(PyUnicode_Check(o)) {
            const char* c = PyUnicode_AsUTF8(o);
            return std::string(c);
        }
        else {
            std::string msg = "could not convert ";
            msg += o->ob_type->tp_name;
            msg += " to string";
            throw std::domain_error(msg);
        }
    }

    template<>
    int16_t cast(PyObject *o) {return (int16_t)cast<PyObject, int>(o);}

    template<>
    uint16_t cast(PyObject *o) {return (uint16_t)cast<PyObject, int>(o);}

    template<>
    uint32_t cast(PyObject *o) {return (uint32_t)cast<PyObject, int>(o);}

    template<>
    uint64_t cast(PyObject *o) {return (uint64_t)cast<PyObject, int>(o);}

    template <>
    bool check<std::string>(PyObject *o) {
        MxCHECKPYRET(0)

        return o && PyUnicode_Check(o);
    }

    template <>
    bool check<float>(PyObject *o) {
        MxCHECKPYRET(0)

        return o && PyNumber_Check(o);
    }

    std::string repr(PyObject *o) {
        MxCHECKPYRET("")

        PyObject* pRepr = PyObject_Repr( o ) ;
        
        PyObject * str=PyUnicode_AsASCIIString(pRepr);
        std::string result = std::string(PyBytes_AsString(str));
        Py_DECREF(str);
        
        Py_DecRef( pRepr ) ;
        return result;
    }

    std::string str(PyObject *o) {
        MxCHECKPYRET("")

        std::string result;
        if(o) {
            PyObject* pStr = PyObject_Str( o ) ;
            if(pStr) {
                PyObject *str = PyUnicode_AsASCIIString(pStr);
                result = std::string(PyBytes_AsString(str));
                Py_DECREF(str);
                Py_DecRef( pStr ) ;
            }
            else {
                result += "error calling PyObject_Str(o)";
            }
        }
        else {
            result = "NULL";
        }
        return result;
    }

}
