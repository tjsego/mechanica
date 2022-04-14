/**
 * @file MxCStyle.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxStyle
 * @date 2022-03-30
 */

#ifndef _WRAPS_C_MXCSTYLE_H_
#define _WRAPS_C_MXCSTYLE_H_

#include <mx_port.h>

#include "MxCParticle.h"

// Handles

/**
 * @brief Handle to a @ref MxStyle instance
 * 
 */
struct CAPI_EXPORT MxStyleHandle {
    void *MxObj;
};


/////////////
// MxStyle //
/////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_init(struct MxStyleHandle *handle);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param color 3-element RGB color array
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_initC(struct MxStyleHandle *handle, float *color, bool visible);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param colorName name of a color
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_initS(struct MxStyleHandle *handle, const char *colorName, bool visible);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_destroy(struct MxStyleHandle *handle);

/**
 * @brief Set the color by name
 * 
 * @param handle populated handle
 * @param colorName name of color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_setColor(struct MxStyleHandle *handle, const char *colorName);

/**
 * @brief Test whether the instance is visible
 * 
 * @param handle populated handle
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_getVisible(struct MxStyleHandle *handle, bool *visible);

/**
 * @brief Set whether the instance is visible
 * 
 * @param handle populated handle
 * @param visible visible flag
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCStyle_setVisible(struct MxStyleHandle *handle, bool visible);

/**
 * @brief Construct and apply a new color map for a particle type and species
 * 
 * @param partType particle type
 * @param speciesName name of species
 * @param name name of color map
 * @param min minimum value of map
 * @param max maximum value of map
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStyle_newColorMapper(struct MxStyleHandle *handle, 
                                           struct MxParticleTypeHandle *partType, 
                                           const char *speciesName, 
                                           const char *name, 
                                           float min, 
                                           float max);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStyle_toString(struct MxStyleHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCStyle_fromString(struct MxStyleHandle *handle, const char *str);

#endif // _WRAPS_C_MXCSTYLE_H_