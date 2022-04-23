/**
 * @file mechanica_c.c
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica C API
 * @date 2022-03-24
 */

#include "mechanica_c.h"

#include "mechanica_c_private.h"

#include <mx_config.h>


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCVersionStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_VERSION, str, numChars);
}

HRESULT MxCSystemNameStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_SYSTEM_NAME, str, numChars);
}

HRESULT MxCSystemVersionStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_SYSTEM_VERSION, str, numChars);
}

HRESULT MxCCompilerIDStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_COMPILER_ID, str, numChars);
}

HRESULT MxCCompilerVersionStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MX_COMPILER_VERSION, str, numChars);
}

HRESULT MxCBuildDateStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(mxBuildDate(), str, numChars);
}

HRESULT MxCBuildTimeStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(mxBuildTime(), str, numChars);
}

HRESULT MxCVersionMajorStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(std::to_string(MX_VERSION_MAJOR), str, numChars);
}

HRESULT MxCVersionMinorStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(std::to_string(MX_VERSION_MINOR), str, numChars);
}

HRESULT MxCVersionPatchStr(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(std::to_string(MX_VERSION_PATCH), str, numChars);
}

bool MxChasCUDA() {
    return mxHasCuda();
}
