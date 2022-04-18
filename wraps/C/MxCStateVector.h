/**
 * @file MxCStateVector.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxStateVector
 * @date 2022-04-01
 */

#ifndef _WRAPS_C_MXCSTATEVECTOR_H_
#define _WRAPS_C_MXCSTATEVECTOR_H_

#include <mx_port.h>

// Handles

/**
 * @brief Handle to a @ref MxStateVector instance
 * 
 */
struct CAPI_EXPORT MxStateVectorHandle {
    void *MxObj;
};


///////////////////
// MxStateVector //
///////////////////


/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_destroy(struct MxStateVectorHandle *handle);

/**
 * @brief Get size of vector
 * 
 * @param handle populated handle
 * @param size vector size
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_getSize(struct MxStateVectorHandle *handle, unsigned int *size);

/**
 * @brief Get the species of the state vector
 * 
 * @param handle populated handle
 * @param slist species list
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStateVector_getSpecies(struct MxStateVectorHandle *handle, struct MxSpeciesListHandle *slist);

/**
 * @brief reset the species values based on the values specified in the species.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_reset(struct MxStateVectorHandle *handle);

/**
 * @brief Get a summary string of the state vector
 * 
 * @param handle populated handle
 * @param str array to populate
 * @param numChars number of array characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_getStr(struct MxStateVectorHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Get the value of an item
 * 
 * @param handle populated handle
 * @param i index of item
 * @param value value of item
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_getItem(struct MxStateVectorHandle *handle, int i, float *value);

/**
 * @brief Set the value of an item
 * 
 * @param handle populated handle
 * @param i index of item
 * @param value value of item
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_setItem(struct MxStateVectorHandle *handle, int i, float value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_toString(struct MxStateVectorHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStateVector_fromString(struct MxStateVectorHandle *handle, const char *str);

#endif // _WRAPS_C_MXCSTATEVECTOR_H_