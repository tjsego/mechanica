/**
 * @file MxSpeciesValue.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines wrap of valued SBML species; derived from carbon CSpeciesValue.hpp written by Andy Somogyi
 * @date 2021-07-09
 * 
 */

#ifndef SRC_STATE_MXSPECIESVALUE_H_
#define SRC_STATE_MXSPECIESVALUE_H_

#include <mx_port.h>
#include "MxSpecies.h"

struct MxSpeciesValue
{
    double value;
    struct MxStateVector *state_vector;
    uint32_t index;

    MxSpecies *species();

    bool getBoundaryCondition();
    double getInitialAmount();
    double getInitialConcentration();
    bool getConstant();
    double secrete(const double &amount, struct MxParticleList *to=NULL, double *distance=NULL);

    MxSpeciesValue(const double &value);
    MxSpeciesValue(const double &value, struct MxStateVector *state_vector, uint32_t index);
};

#endif /* SRC_STATE_MXSPECIESVALUE_H_ */