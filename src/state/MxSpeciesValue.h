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

/**
 * @brief A working valued-object of an underlying MxSpecies attached to an object. 
 */
struct MxSpeciesValue
{
    double value;
    struct MxStateVector *state_vector;
    uint32_t index;

    MxSpecies *species();

    bool getBoundaryCondition();
    int setBoundaryCondition(const int &value);
    double getInitialAmount();
    int setInitialAmount(const double &value);
    double getInitialConcentration();
    int setInitialConcentration(const double &value);
    bool getConstant();
    int setConstant(const int &value);

    /**
     * @brief Secrete this species into a neighborhood. 
     * 
     * Requires either a list of neighboring particles or neighborhood distance. 
     * 
     * @param amount Amount to secrete. 
     * @param to Optional list of particles to secrete to. 
     * @param distance Neighborhood distance. 
     * @return double Amount actually secreted, accounting for availability and other subtleties. 
     */
    double secrete(const double &amount, struct MxParticleList *to=NULL, double *distance=NULL);

    MxSpeciesValue(const double &value);
    MxSpeciesValue(const double &value, struct MxStateVector *state_vector, uint32_t index);
};

#endif /* SRC_STATE_MXSPECIESVALUE_H_ */