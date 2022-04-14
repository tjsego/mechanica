/**
 * @file MxCLogger.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxLogger
 * @date 2022-04-04
 */

#include "MxCLogger.h"

#include "mechanica_c_private.h"

#include <MxLogger.h>


////////////////
// MxLogLevel //
////////////////


HRESULT MxCLogLevel_init(struct MxLogLevelHandle *handle) {
    MXCPTRCHECK(handle);
    handle->LOG_CURRENT = LOG_CURRENT;
    handle->LOG_FATAL = LOG_FATAL;
    handle->LOG_CRITICAL = LOG_CRITICAL;
    handle->LOG_ERROR = LOG_ERROR;
    handle->LOG_WARNING = LOG_WARNING;
    handle->LOG_NOTICE = LOG_NOTICE;
    handle->LOG_INFORMATION = LOG_INFORMATION;
    handle->LOG_DEBUG = LOG_DEBUG;
    handle->LOG_TRACE = LOG_TRACE;
    return S_OK;
}


////////////////
// MxLogEvent //
////////////////


HRESULT MxCLogEventHandle_init(struct MxLogEventHandle *handle) {
    MXCPTRCHECK(handle);
    handle->LOG_OUTPUTSTREAM_CHANGED = LOG_OUTPUTSTREAM_CHANGED;
    handle->LOG_LEVEL_CHANGED = LOG_LEVEL_CHANGED;
    handle->LOG_CALLBACK_SET = LOG_CALLBACK_SET;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCLogger_setLevel(unsigned int level) {
    MxLogger::setLevel(level);
    return S_OK;
}

HRESULT MxCLogger_getLevel(unsigned int *level) {
    MXCPTRCHECK(level);
    *level = MxLogger::getLevel();
    return S_OK;
}

HRESULT MxCLogger_enableFileLogging(const char *fileName, unsigned int level) {
    MxLogger::enableFileLogging(fileName, level);
    return S_OK;
}

HRESULT MxCLogger_disableFileLogging() {
    MxLogger::disableFileLogging();
    return S_OK;
}

HRESULT MxCLogger_getFileName(char **str, unsigned int *numChars) {
    return mx::capi::str2Char(MxLogger::getFileName(), str, numChars);
}

HRESULT MxCLogger_log(unsigned int level, const char *msg) {
    MxLogger::log((MxLogLevel)level, msg);
    return S_OK;
}
