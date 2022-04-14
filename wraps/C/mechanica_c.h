/**
 * @file mechanica_c.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the Mechanica C API
 * @date 2022-03-24
 */

#ifndef _WRAPS_C_MECHANICA_C_H_
#define _WRAPS_C_MECHANICA_C_H_

#include <Mechanica.h>


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Mechanica version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCVersionStr(char **str, unsigned int *numChars);

/**
 * @brief System name
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystemNameStr(char **str, unsigned int *numChars);

/**
 * @brief System version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystemVersionStr(char **str, unsigned int *numChars);

/**
 * @brief Package compiler ID
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCompilerIDStr(char **str, unsigned int *numChars);

/**
 * @brief Package compiler version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCCompilerVersionStr(char **str, unsigned int *numChars);

/**
 * @brief Package build date
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBuildDateStr(char **str, unsigned int *numChars);

/**
 * @brief Package build time
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBuildTimeStr(char **str, unsigned int *numChars);

/**
 * @brief Mechanica major version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCVersionMajorStr(char **str, unsigned int *numChars);

/**
 * @brief Mechanica minor version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCVersionMinorStr(char **str, unsigned int *numChars);

/**
 * @brief Mechanica patch version
 * 
 * @param str value
 * @param numChars number of characters
 * @return S_OK on success  
 */
CAPI_FUNC(HRESULT) MxCVersionPatchStr(char **str, unsigned int *numChars);

/**
 * @brief Test whether the installation supports CUDA
 * 
 * @return true if CUDA is supported
 */
CAPI_FUNC(bool) MxChasCUDA();


#endif // _WRAPS_C_MECHANICA_C_H_