/*
 * Flux.h
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_FLUX_H_
#define SRC_MDCORE_SRC_FLUX_H_

#include "platform.h"
#include "mdcore_single_config.h"
#include "space_cell.h"

#include <string>

enum FluxKind {
    FLUX_FICK = 0,
    FLUX_SECRETE = 1,
    FLUX_UPTAKE = 2
};


// keep track of the ids of the particle types, to determine
// the reaction direction.
struct TypeIdPair {
    int16_t a;
    int16_t b;
};

struct MxFlux {
    int32_t       size; // temporary size until we get SIMD instructions.
    int8_t        kinds[MX_SIMD_SIZE];
    TypeIdPair    type_ids[MX_SIMD_SIZE];
    int32_t       indices_a[MX_SIMD_SIZE];
    int32_t       indices_b[MX_SIMD_SIZE];
    float         coef[MX_SIMD_SIZE];
    float         decay_coef[MX_SIMD_SIZE];
    float         target[MX_SIMD_SIZE];
};

struct MxParticleType;

/**
 * @brief A flux is defined between a pair of types, and acts on the
 * state vector between a pair of instances.
 *
 * The indices of the species in each state vector
 * are most likely different, so Mechanica tracks the
 * indices in each type, and the transport constatants.
 *
 * A flux between a pair of types, and pair of respective
 * species needs:
 *
 * (1) type A, (2) type B, (3) species id in A, (4) species id in B,
 * (5) transport constant.
 *
 * Allocates Flux as a single block, member pointers point to
 * offsets in these blocks.
 *
 * Allocated size is:
 * sizeof(MxFluxes) + 2 * alloc_size * sizeof(int32) + alloc_size * sizeof(float)
 */
struct CAPI_EXPORT MxFluxes
{
    int32_t size;          // how many individual flux objects this has
    int32_t fluxes_size;   // how many fluxes (blocks) this has.
    // static int32_t init;
    MxFlux fluxes[];       // allocated in single block, this

    static MxFluxes* newFluxes(int32_t init_size);
    static MxFluxes *create(FluxKind kind, MxParticleType *a, MxParticleType *b,
                            const std::string& name, float k, float decay, float target);
    static MxFluxes *addFlux(FluxKind kind, MxFluxes *fluxes,
                             int16_t typeId_a, int16_t typeId_b,
                             int32_t index_a, int32_t index_b,
                             float k, float decay, float target);

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
     * Automatically updates when running on a CUDA device. 
     * 
     * @param A first type
     * @param B second type
     * @param name name of species
     * @param k transport coefficient
     * @param decay optional decay. Defaults to 0.0. 
     * @return MxFluxes* 
     */
    static MxFluxes *fluxFick(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &decay=0.0f);

    /**
     * @brief Alias of fluxFick. 
     * 
     * @param A first type
     * @param B second type
     * @param name name of species
     * @param k transport coefficient
     * @param decay optional decay. Defaults to 0.0. 
     * @return MxFluxes* 
     */
    static MxFluxes *flux(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &decay=0.0f);

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
     * Automatically updates when running on a CUDA device. 
     * 
     * @param A first type
     * @param B second type
     * @param name name of species
     * @param k transport coefficient
     * @param target target concentration
     * @param decay optional decay. Defaults to 0.0 
     * @return MxFluxes* 
     */
    static MxFluxes *secrete(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &target, const float &decay=0.0f);

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
     * Automatically updates when running on a CUDA device. 
     * 
     * @param A first type
     * @param B second type
     * @param name name of species
     * @param k transport coefficient
     * @param target target concentration
     * @param decay optional decay. Defaults to 0.0 
     * @return MxFluxes* 
     */
    static MxFluxes *uptake(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &target, const float &decay=0.0f);
};

/**
 * integrate all of the fluxes for a space cell.
 */
HRESULT MxFluxes_integrate(space_cell *cell, float dt=-1.0);

/**
 * integrate all of the fluxes for a space cell.
 */
HRESULT MxFluxes_integrate(int cellId);

#endif /* SRC_MDCORE_SRC_FLUX_H_ */
