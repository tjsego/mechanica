/*
 * metrics.h
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_METRICS_H_
#define SRC_MDCORE_INCLUDE_METRICS_H_

#include "platform.h"
#include "mdcore_config.h"
#include "MxParticle.h"
#include <set>
#include <vector>

/**
 * @brief Computes the relative position with respect to an origin while 
 * optionally account for boundary conditions. 
 * 
 * If boundaries along a dimension are periodic, then this chooses the 
 * relative coordinate closest to the origin. 
 * 
 * @param pos absolute position
 * @param origin origin
 * @param comp_bc flag to compensate for boundary conditions; default true
 * @return MxVector3f relative position with respect to the given origin
 */
CPPAPI_FUNC(MxVector3f) MxRelativePosition(const MxVector3f &pos, const MxVector3f &origin, const bool &comp_bc=true);

/**
 * @origin [in] origin of the sphere where we will comptute
 * the local virial tensor.
 * @radius [in] include all partices a given radius in calculation. 
 * @typeIds [in] vector of type ids to indlude in calculation,
 * if empty, includes all particles.
 * @tensor [out] result vector, writes a 3x3 matrix in a row-major in the given
 * location.
 *
 * If periodoc, we don't include the periodic image cells, because we only
 * calculate the forces within the simulation volume.
 */
CPPAPI_FUNC(HRESULT) MxCalculateVirial(FPTYPE *origin,
                                       FPTYPE radius,
                                       const std::set<short int> &typeIds,
                                       FPTYPE *tensor);

/**
 * calculate the virial tensor for a specific list of particles.
 * currently uses center of mass as origin, may change in the
 * future with different flags.
 *
 * flags currently ignored.
 */
CAPI_FUNC(HRESULT) MxParticles_Virial(int32_t *parts,
                                        uint16_t nr_parts,
                                        uint32_t flags,
                                        FPTYPE *tensor);

/**
*
* flags currently ignored.
*/
CAPI_FUNC(HRESULT) MxParticles_StressTensor(int32_t *parts,
                                       uint16_t nr_parts,
                                       uint32_t flags,
                                       FPTYPE *tensor);


/**
 * @param result: pointer to float to store result.
 */
CAPI_FUNC(HRESULT) MxParticles_RadiusOfGyration(int32_t *parts, uint16_t nr_parts, float* result);

/**
 * @param result: pointer to float[3] to store result
 */
CAPI_FUNC(HRESULT) MxParticles_CenterOfMass(int32_t *parts, uint16_t nr_parts, float* result);

/**
 * @param result: pointer to float[3] to store result.
 */
CAPI_FUNC(HRESULT) MxParticles_CenterOfGeometry(int32_t *parts, uint16_t nr_parts, float* result);

/**
 * @param result: pointer to float[9] to store result.
 */
CAPI_FUNC(HRESULT) MxParticles_MomentOfInertia(int32_t *parts, uint16_t nr_parts, float* result);

/**
 * converts cartesian to spherical, writes spherical
 * coords in to result array.
 * return MxVector3f{radius, theta, phi};
 */
CPPAPI_FUNC(MxVector3f) MxCartesianToSpherical(const MxVector3f& postion, const MxVector3f& origin);

/**
 * Searches and enumerates a location of space for all particles there.
 *
 * Allocates a buffer, and stores the results there.
 *
 * @param typeIds [optional] set of type ids to include. If not given,
 * gets all other parts within radius.
 * 
 * @param nr_parts, out, number of parts
 * @param parts, out, newly allocated buffer of particle ids.
 */
CAPI_FUNC(HRESULT) MxParticle_Neighbors(struct MxParticle *part,
                                          FPTYPE radius,
                                          const std::set<short int> *typeIds,
                                          uint16_t *nr_parts,
                                          int32_t **parts);


/**
 * Creates an array of ParticleList objects.
 */
std::vector<std::vector<std::vector<MxParticleList*> > > MxParticle_Grid(const MxVector3i &shape);

CAPI_FUNC(HRESULT) MxParticle_Grid(const MxVector3i &shape, MxParticleList **result);


#endif /* SRC_MDCORE_INCLUDE_METRICS_H_ */
