/**
 * @file MxBoundaryConditionsPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxBoundaryConditions
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXBOUNDARYCONDITIONSPY_H_
#define _SRC_LANGS_PY_MXBOUNDARYCONDITIONSPY_H_

#include "MxPy.h"
#include <MxBoundaryConditions.hpp>

struct MxBoundaryConditionsArgsContainerPy : MxBoundaryConditionsArgsContainer {

    MxBoundaryConditionsArgsContainerPy(PyObject *obj);

};

#endif // _SRC_LANGS_PY_MXBOUNDARYCONDITIONSPY_H_
