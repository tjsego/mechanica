/**
 * @file MxPotentialPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxPotential
 * @date 2022-03-23
 * 
 */

#include "MxPotentialPy.h"


static double pyEval(PyObject *f, double r) {
	PyObject *py_r = mx::cast<double, PyObject*>(r);
	PyObject *args = PyTuple_Pack(1, py_r);
	PyObject *py_result = PyObject_CallObject(f, args);
	Py_XDECREF(args);

	if (py_result == NULL) {
		PyObject *err = PyErr_Occurred();
		PyErr_Clear();
		return 0.0;
	}

	double result = mx::cast<PyObject, double>(py_result);
	Py_DECREF(py_result);
	return result;
}

static PyObject *pyCustom_f, *pyCustom_fp, *pyCustom_f6p;

static double pyEval_f(double r) {
	return pyEval(pyCustom_f, r);
}

static double pyEval_fp(double r) {
	return pyEval(pyCustom_fp, r);
}

static double pyEval_f6p(double r) {
	return pyEval(pyCustom_f6p, r);
}

MxPotential *MxPotentialPy::customPy(double min, double max, PyObject *f, PyObject *fp, PyObject *f6p, double *tol, uint32_t *flags) {
	pyCustom_f = f;
	double (*eval_fp)(double) = NULL;
	double (*eval_f6p)(double) = NULL;

	if (fp != Py_None) {
		pyCustom_fp = fp;
		eval_fp = &pyEval_fp;
	}
	
	if (f6p != Py_None) {
		pyCustom_f6p = f6p;
		eval_f6p = &pyEval_f6p;
	}

	auto p = MxPotential::custom(min, max, &pyEval_f, eval_fp, eval_f6p, tol, flags);

	pyCustom_f = NULL;
	pyCustom_fp = NULL;
	pyCustom_f6p = NULL;

	return p;
}
