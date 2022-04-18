/**
 * @file MxApplicationPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxApplication
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXAPPLICATIONPY_H_
#define _SRC_LANGS_PY_MXAPPLICATIONPY_H_

#include "MxPy.h"

#include <rendering/MxApplication.h>


PyObject* MxTestImage(PyObject* dummyo);
PyObject* MxFramebufferImageData(PyObject *dummyo);


#endif // _SRC_LANGS_PY_MXAPPLICATIONPY_H_