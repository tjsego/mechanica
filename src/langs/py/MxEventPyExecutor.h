/**
 * @file MxEventPyExecutor.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines template for implementing callbacks in python
 * @date 2021-07-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXEVENTPYEXECUTOR_H_
#define _SRC_LANGS_PY_MXEVENTPYEXECUTOR_H_

#include "MxPy.h"


template<typename event_t>
struct MxEventPyExecutor {

    /**
     * @brief Issues call to execute event callback in python layer on existing event
     * 
     * @return HRESULT 
     */
    HRESULT invoke() {
        if(!hasExecutorPyCallable() || !activeEvent) return E_ABORT;

        PyObject *result = PyObject_CallObject(executorPyCallable, NULL);

        if(result == NULL) {
            PyObject *err = PyErr_Occurred();
            PyErr_Clear();
            return E_FAIL;
        }
        Py_DECREF(result);

        return S_OK;
    }

    /**
     * @brief Issues call to execute event callback in python layer on new event
     * 
     * @param ke event on which to execute callback
     * @return HRESULT 
     */
    HRESULT invoke(event_t &ke) {
        activeEvent = &ke;

        return invoke();
    }

    /**
     * @brief Gets the current event object
     * 
     * @return event_t* 
     */
    event_t *getEvent() {
        return activeEvent;
    }

    /**
     * @brief Tests whether the executor callback from the python layer has been set
     * 
     * @return true callback has been set
     * @return false callback has not been set
     */
    bool hasExecutorPyCallable() { return executorPyCallable != NULL; }

    /**
     * @brief Sets the executor callback from the python layer
     * 
     * @param callable executor callback from the python layer
     */
    void setExecutorPyCallable(PyObject *callable) {
        resetExecutorPyCallable();
        executorPyCallable = callable;
        Py_INCREF(callable);
    }

    /**
     * @brief Sets the executor callback from the python layer if it has not yet been set
     * 
     * @param callable executor callback from the python layer
     */
    void maybeSetExecutorPyCallable(PyObject *callable) {
        if(hasExecutorPyCallable()) return;
        executorPyCallable = callable;
        Py_INCREF(callable);
    }

    /**
     * @brief Resets the executor callback from the python layer
     * 
     */
    void resetExecutorPyCallable() {
        if(hasExecutorPyCallable()) Py_DECREF(executorPyCallable);
        executorPyCallable = NULL;
    }

protected:

    // The current event object, if any
    event_t *activeEvent = NULL;

    // The executor callback from the python layer
    PyObject *executorPyCallable = NULL;

};

#endif // _SRC_LANGS_PY_MXEVENTPYEXECUTOR_H_