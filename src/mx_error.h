/*
 * mx_error.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MX_ERROR_H_
#define SRC_MX_ERROR_H_

#include <mx_port.h>

#include <exception>

struct CAPI_EXPORT MxError {
    HRESULT err;
    const char* msg;
    int lineno;
    const char* fname;
    const char* func;
};

enum MxError_Options {
};

#define mx_error(code, msg) MxErr_Set(code, msg, __LINE__, __FILE__, MX_FUNCTION)

#define mx_exp(e) MxExp_Set(e, "", __LINE__, __FILE__, MX_FUNCTION)

#define MX_RETURN_EXP(e) MxExp_Set(e, "", __LINE__, __FILE__, MX_FUNCTION); return NULL

CAPI_FUNC(HRESULT) MxErr_Set(HRESULT code, const char* msg, int line, const char* file, const char* func);

CAPI_FUNC(HRESULT) MxExp_Set(const std::exception&, const char* msg, int line, const char* file, const char* func);

CAPI_FUNC(MxError*) MxErr_Occurred();

// CAPI_FUNC(void) MxErr_SetOptions(uint32_t options);

/**
 * Clear the error indicator. If the error indicator is not set, there is no effect.
 */
CAPI_FUNC(void) MxErr_Clear();

#endif /* SRC_MX_ERROR_H_ */
