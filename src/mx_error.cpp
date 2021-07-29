/*
 * mx_error.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <cstdio>

#include <mx_error.h>

#include <iostream>
#include <sstream>
#include <MxLogger.h>

static MxError Error;
static MxError *ErrorPtr = NULL;

CAPI_FUNC(HRESULT) MxErr_Set(HRESULT code, const char* msg, int line,
		const char* file, const char* func) {

	Error.err = code;
	Error.fname = file;
	Error.func = func;
	Error.msg = msg;
    
    std::string logstr = "Code: ";
    logstr += std::to_string(code);
    logstr += ", Msg: ";
    logstr += msg;
    logstr += ", File: " + std::string(file);
    logstr += ", Line: " + std::to_string(line);
    logstr += ", Function: " + std::string(func);
    MxLogger::log(LOG_ERROR, logstr);

	ErrorPtr = &Error;
	return code;
}

CAPI_FUNC(HRESULT) MxExp_Set(const std::exception& e, const char* msg, int line, const char* file, const char* func) {
    std::cerr << "error: " << e.what() << ", " << msg << ", " << line << ", " << func << std::endl;
    
    Error.err = E_FAIL;
    Error.fname = file;
    Error.func = func;
    Error.msg = msg;
    
    ErrorPtr = &Error;
    return E_FAIL;
}

CAPI_FUNC(MxError*) MxErr_Occurred() {
    return ErrorPtr;
}

CAPI_FUNC(void) MxErr_Clear() {
    ErrorPtr = NULL;
}

// CAPI_FUNC(void) MxErr_SetOptions(uint32_t options) {
//     MxError_Opt = options;
// }
