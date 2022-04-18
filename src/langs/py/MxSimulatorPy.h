/**
 * @file MxSimulatorPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxSimulator
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXSIMULATORPY_H_
#define _SRC_LANGS_PY_MXSIMULATORPY_H_

#include "MxPy.h"

#include <MxSimulator.h>


struct CAPI_EXPORT MxSimulatorPy : MxSimulator {

public:

    /**
     * gets the global simulator object, throws exception if fail.
     */
    static MxSimulatorPy *get();

    static PyObject *_run(PyObject *args, PyObject *kwargs);
    
    /**
     * @brief Interactive python version of the run loop. This checks the ipython context and lets 
     * ipython process keyboard input, while we also running the simulator and processing window messages.
     * 
     * @return HRESULT 
     */
    static HRESULT irun();

    static HRESULT _show();

    static void *wait_events(const double &timeout=-1);

};

CAPI_FUNC(HRESULT) _setIPythonInputHook(PyObject *_ih);

CAPI_FUNC(HRESULT) _onIPythonNotReady();

/**
 * @brief Initialize a simulation in Python
 * 
 * @param args positional arguments; first argument is name of simulation (if any)
 * @param kwargs keyword arguments; currently supported are
 * 
 *      dim: (3-component list of floats) the dimensions of the spatial domain; default is [10., 10., 10.]
 * 
 *      cutoff: (float) simulation cutoff distance; default is 1.
 * 
 *      cells: (3-component list of ints) the discretization of the spatial domain; default is [4, 4, 4]
 * 
 *      threads: (int) number of threads; default is hardware maximum
 * 
 *      integrator: (int) simulation integrator; default is FORWARD_EULER
 * 
 *      dt: (float) time discretization; default is 0.01
 * 
 *      bc: (int or dict) boundary conditions; default is everywhere periodic
 * 
 *      window_size: (2-component list of ints) size of application window; default is [800, 600]
 * 
 *      seed: (int) seed for pseudo-random number generator
 * 
 *      load_file: (str) path to saved simulation state to initialize
 * 
 *      logger_level: (int) logger level; default is no logging
 * 
 *      clip_planes: (list of tuple of (MxVector3f, MxVector3f)) list of point-normal pairs of clip planes; default is no planes
 */
CAPI_FUNC(PyObject *) MxSimulatorPy_init(PyObject *args, PyObject *kwargs);

#endif // _SRC_LANGS_PY_MXSIMULATORPY_H_