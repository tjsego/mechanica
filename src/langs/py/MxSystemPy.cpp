/**
 * @file MxSystemPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxSystem
 * @date 2022-03-23
 * 
 */

#include "MxSystemPy.h"

#include "MxApplicationPy.h"


PyObject *MxSystemPy::test_image() {
    return MxTestImage(Py_None);
}

PyObject *MxSystemPy::image_data() {
    return MxFramebufferImageData(Py_None);
}

bool MxSystemPy::is_terminal_interactive() {
    return Mx_TerminalInteractiveShell();
}

bool MxSystemPy::is_jupyter_notebook() {
    return MxPy_ZMQInteractiveShell();
}

PyObject *MxSystemPy::jwidget_init(PyObject *args, PyObject *kwargs) {
    
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
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
        mx_error(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* init = PyObject_GetAttrString(module,(char*)"init");
    
    if(!init) {
        mx_error(E_FAIL, "mechanica.jwidget package does not have an init function");
        return NULL;
    }

    PyObject* result = PyObject_Call(init, args, kwargs);
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(init);
    
    if(!result) {
        Log(LOG_ERROR) << "error calling mechanica.jwidget.init: " << mx::pyerror_str();
    }
    
    return result;
}

PyObject *MxSystemPy::jwidget_run(PyObject *args, PyObject *kwargs) {
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
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
        mx_error(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* run = PyObject_GetAttrString(module,(char*)"run");
    
    if(!run) {
        mx_error(E_FAIL, "mechanica.jwidget package does not have an run function");
        return NULL;
    }

    PyObject* result = PyObject_Call(run, args, kwargs);

    if (!result) {
        Log(LOG_ERROR) << "error calling mechanica.jwidget.run: " << mx::pyerror_str();
    }

    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(run);
    
    return result;
    
}
