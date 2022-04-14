/**
 * @file MxCFlux.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxFluxes
 * @date 2022-04-03
 */

#ifndef _WRAPS_C_MXCFLUX_H_
#define _WRAPS_C_MXCFLUX_H_

#include <mx_port.h>

#include "MxCParticle.h"

// Handles

struct CAPI_EXPORT MxFluxKindHandle {
    unsigned int FLUX_FICK;
    unsigned int FLUX_SECRETE;
    unsigned int FLUX_UPTAKE;
};

/**
 * @brief Handle to a @ref MxFlux instance
 * 
 */
struct CAPI_EXPORT MxFluxHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxFluxes instance
 * 
 */
struct CAPI_EXPORT MxFluxesHandle {
    void *MxObj;
};


//////////////
// FluxKind //
//////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCFluxKindHandle_init(struct MxFluxKindHandle *handle);


////////////
// MxFlux //
////////////


/**
 * @brief Get the size of the fluxes
 * 
 * @param handle populated handle
 * @param size flux size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getSize(struct MxFluxHandle *handle, unsigned int *size);

/**
 * @brief Get the kind of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param kind flux kind
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getKind(struct MxFluxHandle *handle, unsigned int index, unsigned int *kind);

/**
 * @brief Get the type ids of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param typeid_a id of first type
 * @param typeid_b id of second type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getTypeIds(struct MxFluxHandle *handle, unsigned int index, unsigned int *typeid_a, unsigned int *typeid_b);

/**
 * @brief Get the coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param coef flux coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getCoef(struct MxFluxHandle *handle, unsigned int index, float *coef);

/**
 * @brief Set the coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param coef flux coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_setCoef(struct MxFluxHandle *handle, unsigned int index, float coef);

/**
 * @brief Get the decay coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param decay_coef flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getDecayCoef(struct MxFluxHandle *handle, unsigned int index, float *decay_coef);

/**
 * @brief Set the decay coefficient of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param decay_coef flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_setDecayCoef(struct MxFluxHandle *handle, unsigned int index, float decay_coef);

/**
 * @brief Get the target of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param target flux target
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_getTarget(struct MxFluxHandle *handle, unsigned int index, float *target);

/**
 * @brief Set the target of a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param target flux target
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFlux_setTarget(struct MxFluxHandle *handle, unsigned int index, float target);


//////////////
// MxFluxes //
//////////////


/**
 * @brief Get the number of individual flux objects
 * 
 * @param handle populated handle
 * @param size number of individual flux objects
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFluxes_getSize(struct MxFluxesHandle *handle, int *size);

/**
 * @brief Get a flux
 * 
 * @param handle populated handle
 * @param index flux index
 * @param flux flux
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFluxes_getFlux(struct MxFluxesHandle *handle, unsigned int index, struct MxFluxHandle *flux);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Creates and binds a Fickian diffusion flux. 
 * 
 * Fickian diffusion flux implements the analogous reaction: 
 * 
 * @f[
 *      a.S \leftrightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}} \right)\left(a.S - b.S\right) , 
 * @f]
 * 
 * @f[
 *      a.S \rightarrow 0   ; \frac{d}{2} a.S , 
 * @f]
 * 
 * @f[
 *      b.S \rightarrow 0   ; \frac{d}{2} b.S , 
 * @f]
 * 
 * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
 * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
 * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
 * distance, and @f$ d @f$ is an optional decay term. 
 * 
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFluxes_fluxFick(struct MxFluxesHandle *handle, 
                                      struct MxParticleTypeHandle *A, 
                                      struct MxParticleTypeHandle *B, 
                                      const char *name, 
                                      float k, 
                                      float decay);

/**
 * @brief Creates a secretion flux by active pumping. 
 * 
 * Secretion flux implements the analogous reaction: 
 * 
 * @f[
 *      a.S \rightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}} \right)\left(a.S - a.S_{target} \right) ,
 * @f]
 * 
 * @f[
 *      a.S \rightarrow 0   ; \frac{d}{2} a.S ,
 * @f]
 * 
 * @f[
 *      b.S \rightarrow 0   ; \frac{d}{2} b.S ,
 * @f]
 * 
 * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
 * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
 * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
 * distance, and @f$ d @f$ is an optional decay term. 
 * 
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param target flux target
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFluxes_secrete(struct MxFluxesHandle *handle, 
                                     struct MxParticleTypeHandle *A, 
                                     struct MxParticleTypeHandle *B, 
                                     const char *name, 
                                     float k, 
                                     float target, 
                                     float decay);

/**
 * @brief Creates an uptake flux by active pumping. 
 * 
 * Uptake flux implements the analogous reaction: 
 * 
 * @f[
 *      a.S \rightarrow b.S ; k \left(1 - \frac{r}{r_{cutoff}}\right)\left(b.S - b.S_{target} \right)\left(a.S\right) ,
 * @f]
 * 
 * @f[
 *      a.S \rightarrow 0   ; \frac{d}{2} a.S ,
 * @f]
 * 
 * @f[
 *      b.S \rightarrow 0   ; \frac{d}{2} b.S ,
 * @f]
 * 
 * where @f$ a.S @f$ is a chemical species located at object @f$ a @f$, and likewise 
 * for @f$ b @f$, @f$ k @f$ is the flux constant, @f$ r @f$ is the 
 * distance between the two objects, @f$ r_{cutoff} @f$ is the global cutoff 
 * distance, and @f$ d @f$ is an optional decay term. 
 * 
 * @param handle handle to populate
 * @param A first type
 * @param B second type
 * @param name name of species
 * @param k flux transport coefficient
 * @param target flux target
 * @param decay flux decay coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCFluxes_uptake(struct MxFluxesHandle *handle, 
                                    struct MxParticleTypeHandle *A, 
                                    struct MxParticleTypeHandle *B, 
                                    const char *name, 
                                    float k, 
                                    float target, 
                                    float decay);

#endif //_WRAPS_C_MXCFLUX_H_