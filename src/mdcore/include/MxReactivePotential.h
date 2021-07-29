/*
 * MxReactiveBond.h
 *
 *  Created on: Jul 16, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_
#define SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_

#include "MxPotential.h"

typedef HRESULT (*reactivepotential_invoke)(struct MxReactivePotential *,
        struct MxParticle *a, struct MxParticle *b);

/** The #potential structure. */
typedef struct MxReactivePotential : MxPotential {
    double activation_energy;

    double activation_distance;


    /**
     * Bond potential, this is what gets copied to the bond,
     * not used is non-bonded potential.
     */
    MxPotential *bond_potential;

    /**
     * Function that gets invoked when the potential is triggered
     * (energy exceeds threshold).
     */
    reactivepotential_invoke invoke;

} MxReactivePotential;

#endif /* SRC_MDCORE_INCLUDE_MXBONDGENERATOR_H_ */
