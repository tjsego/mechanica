/**
 * @file MxCBind.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBind
 * @date 2022-04-04
 */

#ifndef _WRAPS_C_MXCBIND_H_
#define _WRAPS_C_MXCBIND_H_

#include <mx_port.h>

#include "MxCBond.h"
#include "MxCBoundaryConditions.h"
#include "MxCForce.h"
#include "MxCParticle.h"
#include "MxCPotential.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Bind a potential to a pair of particles
 * 
 * @param p The potential
 * @param a The first particle
 * @param b The second particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_particles(struct MxPotentialHandle *p, struct MxParticleHandleHandle *a, struct MxParticleHandleHandle *b);

/**
 * @brief Bind a potential to a pair of particle types. 
 * 
 * @param p The potential
 * @param a The first type
 * @param b The second type
 * @param bound Flag signifying whether this potential exclusively operates on particles of different clusters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_types(struct MxPotentialHandle *p, struct MxParticleTypeHandle *a, struct MxParticleTypeHandle *b, bool bound);

/**
 * @brief Bind a potential to a pair of particle type and all boundary conditions. 
 * 
 * @param p The potential
 * @param t The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_boundaryConditions(struct MxPotentialHandle *p, struct MxParticleTypeHandle *t);

/**
 * @brief Bind a potential to a pair of particle type and a boundary conditions. 
 * 
 * @param p The potential
 * @param bcs The boundary condition
 * @param t The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_boundaryCondition(struct MxPotentialHandle *p, struct MxBoundaryConditionHandle *bc, struct MxParticleTypeHandle *t);

/**
 * @brief Bind a force to a particle type
 * 
 * @param force The force
 * @param a_type The particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_force(struct MxForceHandle *force, struct MxParticleTypeHandle *a_type);

/**
 * @brief Bind a force to a particle type with magnitude proportional to a species amount
 * 
 * @param force The force
 * @param a_type The particle type
 * @param coupling_symbol The symbol of the species
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_forceS(struct MxForceHandle *force, struct MxParticleTypeHandle *a_type, const char *coupling_symbol);

/**
 * @brief Create bonds for a set of pairs of particles
 * 
 * @param potential The bond potential
 * @param particles The list of particles
 * @param cutoff Interaction cutoff
 * @param ppairsA first elements of type pairs that are bonded, optional
 * @param ppairsB second elements of type pairs that are bonded, optional
 * @param numTypes number of passed type pairs
 * @param half_life Bond half life, optional
 * @param bond_energy Bond dissociation energy, optional
 * @param out List of created bonds, optional
 * @param numOut Number of created bonds, optional
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCBind_bonds(struct MxPotentialHandle *potential,
                                 struct MxParticleListHandle *particles, 
                                 double cutoff, 
                                 struct MxParticleTypeHandle **ppairsA, 
                                 struct MxParticleTypeHandle **ppairsB, 
                                 unsigned int numTypes, 
                                 double *half_life, 
                                 double *bond_energy, 
                                 struct MxBondHandleHandle **out, 
                                 unsigned int *numOut);

#endif // _WRAPS_C_MXCBIND_H_