/*
 * MxSecreteUptake.hpp
 *
 *  Created on: Jan 6, 2021
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_
#define SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_

#include "../../state/MxSpeciesValue.h"
#include "MxParticleList.hpp"

#include <set>

// Simple methods container
struct MxSecreteUptake {

    static double secrete(MxSpeciesValue *species, const double &amount, const MxParticleList &to);
    static double secrete(MxSpeciesValue *species, const double &amount, const double &distance);

};

HRESULT MxSecrete_AmountToParticles(struct MxSpeciesValue* species,
                                    double amount,
                                    uint16_t nr_parts, int32_t *parts,
                                    double *secreted);

HRESULT MxSecrete_AmountWithinDistance(struct MxSpeciesValue* species,
                                       double amount,
                                       double radius,
                                       const std::set<short int> *typeIds,
                                       double *secreted);

#endif /* SRC_MDCORE_SRC_MXSECRETEUPTAKE_HPP_ */
