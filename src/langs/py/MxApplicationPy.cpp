/**
 * @file MxApplicationPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxApplication
 * @date 2022-03-23
 * 
 */


#include "MxApplicationPy.h"

#include <MxLogger.h>


PyObject* MxTestImage(PyObject* dummyo) {

    char *data;
    size_t size;
    std::tie(data, size) = MxTestImage();

    if (data == NULL)
        return NULL;
    
    return PyBytes_FromStringAndSize(data, size);
}

PyObject* MxFramebufferImageData(PyObject *dummyo) {

    Log(LOG_TRACE);
    
    auto jpegData = MxJpegImageData();
    char *data = jpegData.data();
    size_t size = jpegData.size();
    
    return PyBytes_FromStringAndSize(data, size);
}

