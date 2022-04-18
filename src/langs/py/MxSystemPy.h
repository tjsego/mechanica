/**
 * @file MxSystemPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxSystem
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXSYSTEMPY_H_
#define _SRC_LANGS_PY_MXSYSTEMPY_H_

#include "MxPy.h"

#include <MxSystem.h>


struct CAPI_EXPORT MxSystemPy : MxSystem {

public:
   MxSystemPy() {};
   ~MxSystemPy() {};

   static PyObject *test_image();
   static PyObject *image_data();

   /**
    * @brief Test whether Mechanica is running in an interactive terminal
    * 
    * @return true if running in an interactive terminal
    * @return false 
    */
   static bool is_terminal_interactive();

   /**
    * @brief Test whether Mechanica is running in a Jupyter notebook
    * 
    * @return true if running in a Jupyter notebook
    * @return false 
    */
   static bool is_jupyter_notebook();

   static PyObject *jwidget_init(PyObject *args, PyObject *kwargs);
   static PyObject *jwidget_run(PyObject *args, PyObject *kwargs);
};

#endif // _SRC_LANGS_PY_MXSYSTEMPY_H_