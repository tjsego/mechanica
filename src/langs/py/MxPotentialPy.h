/**
 * @file MxPotentialPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxPotential
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXPOTENTIALPY_H_
#define _SRC_LANGS_PY_MXPOTENTIALPY_H_

#include "MxPy.h"
#include <MxPotential.h>


struct MxPotentialPy : MxPotential {
    /**
     * @brief Creates a custom potential. 
     * 
     * @param min The smallest radius for which the potential will be constructed.
     * @param max The largest radius for which the potential will be constructed.
     * @param f function returning the value of the potential
     * @param fp function returning the value of first derivative of the potential
     * @param f6p function returning the value of sixth derivative of the potential
     * @param tol Tolerance, defaults to 0.001.
     * @return MxPotential* 
     */
    static MxPotential *customPy(double min, double max, PyObject *f, PyObject *fp=Py_None, PyObject *f6p=Py_None, 
                                 double *tol=NULL, uint32_t *flags=NULL);
};

#endif // _SRC_LANGS_PY_MXPOTENTIALPY_H_
