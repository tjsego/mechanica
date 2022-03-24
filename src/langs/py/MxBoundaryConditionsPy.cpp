/**
 * @file MxBoundaryConditionsPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxBoundaryConditions
 * @date 2022-03-23
 * 
 */

#include "MxBoundaryConditionsPy.h"

#include <MxLogger.h>

MxBoundaryConditionsArgsContainerPy::MxBoundaryConditionsArgsContainerPy(PyObject *obj) {
    if(PyLong_Check(obj)) setValueAll(mx::cast<PyObject, int>(obj));
    else if(PyDict_Check(obj)) {
        PyObject *keys = PyDict_Keys(obj);

        for(unsigned int i = 0; i < PyList_Size(keys); ++i) {
            PyObject *key = PyList_GetItem(keys, i);
            PyObject *value = PyDict_GetItem(obj, key);

            std::string name = mx::cast<PyObject, std::string>(key);
            if(PyLong_Check(value)) {
                unsigned int v = mx::cast<PyObject, unsigned int>(value);

                Log(LOG_DEBUG) << name << ": " << value;

                setValue(name, v);
            }
            else if(mx::check<std::string>(value)) {
                std::string s = mx::cast<PyObject, std::string>(value);

                Log(LOG_DEBUG) << name << ": " << s;

                setValue(name, MxBoundaryConditions::boundaryKindFromString(s));
            }
            else if(PySequence_Check(value)) {
                std::vector<std::string> kinds;
                PyObject *valueItem;
                for(unsigned int j = 0; j < PySequence_Size(value); j++) {
                    valueItem = PySequence_GetItem(value, j);
                    if(mx::check<std::string>(valueItem)) {
                        std::string s = mx::cast<PyObject, std::string>(valueItem);

                        Log(LOG_DEBUG) << name << ": " << s;

                        kinds.push_back(s);
                    }
                }
                setValue(name, MxBoundaryConditions::boundaryKindFromStrings(kinds));
            }
            else if(PyDict_Check(value)) {
                PyObject *vel = PyDict_GetItemString(value, "velocity");
                if(!vel) {
                    throw std::invalid_argument("attempt to initialize a boundary condition with a "
                                                "dictionary that does not contain a \'velocity\' item, "
                                                "only velocity boundary conditions support dictionary init");
                }
                MxVector3f v = mx::cast<PyObject, MxVector3f>(vel);

                Log(LOG_DEBUG) << name << ": " << v;

                setVelocity(name, v);

                PyObject *restore = PyDict_GetItemString(value, "restore");
                if(restore) {
                    float r = mx::cast<PyObject, float>(restore);

                    Log(LOG_DEBUG) << name << ": " << r;

                    setRestore(name, r);
                }
            }
        }

        Py_DECREF(keys);
    }
}
