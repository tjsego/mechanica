/**
 * @file MxCLogger.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxLogger
 * @date 2022-04-04
 */

#ifndef _WRAPS_C_MXCLOGGER_H_
#define _WRAPS_C_MXCLOGGER_H_

#include <mx_port.h>

// Handles

struct CAPI_EXPORT MxLogLevelHandle {
    unsigned int LOG_CURRENT;
    unsigned int LOG_FATAL;
    unsigned int LOG_CRITICAL;
    unsigned int LOG_ERROR;
    unsigned int LOG_WARNING;
    unsigned int LOG_NOTICE;
    unsigned int LOG_INFORMATION;
    unsigned int LOG_DEBUG;
    unsigned int LOG_TRACE;
};

struct CAPI_EXPORT MxLogEventHandle {
    unsigned int LOG_OUTPUTSTREAM_CHANGED;
    unsigned int LOG_LEVEL_CHANGED;
    unsigned int LOG_CALLBACK_SET;
};


////////////////
// MxLogLevel //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogLevel_init(struct MxLogLevelHandle *handle);


////////////////
// MxLogEvent //
////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCLogEventHandle_init(struct MxLogEventHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Set the Level objectsets the logging level to one a value from Logger::Level
 * 
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_setLevel(unsigned int level);

/**
 * @brief Get the Level objectget the current logging level.
 * 
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_getLevel(unsigned int *level);

/**
 * @brief turns on file logging to the given file as the given level.
 * 
 * @param fileName path to log file
 * @param level logging level
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_enableFileLogging(const char *fileName, unsigned int level);

/**
 * @brief turns off file logging, but has no effect on console logging.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_disableFileLogging();

/**
 * @brief Get the File Name objectget the name of the currently used log file.
 * 
 * @param str string array of file name
 * @param numChars number of characters in string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_getFileName(char **str, unsigned int *numChars);

/**
 * @brief logs a message to the log.
 * 
 * @param level logging level
 * @param msg log message
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCLogger_log(unsigned int level, const char *msg);

#endif // _WRAPS_C_MXCLOGGER_H_