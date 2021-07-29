/**
 * @file MxSpeciesValue.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines wrap of valued SBML species; derived from carbon CSpeciesValue.cpp written by Andy Somogyi
 * @date 2021-07-09
 * 
 */

#include "MxSpeciesValue.h"
#include <state/MxStateVector.h>
#include "MxSpeciesList.h"
#include <mx_error.h>

#include <MxSecreteUptake.hpp>

#include <sbml/Species.h>
#include <iostream>

MxSpecies *MxSpeciesValue::species() {
    if(!(state_vector && state_vector->species)) return NULL;
    return state_vector->species->item(index);
}

MxSpeciesValue::MxSpeciesValue(const double &value) : value(value) {}

MxSpeciesValue::MxSpeciesValue(const double &value, struct MxStateVector *state_vector, uint32_t index) : 
    value(value), state_vector(state_vector), index(index) 
{}

bool MxSpeciesValue::getBoundaryCondition() {
    return species()->getBoundaryCondition();
}

double MxSpeciesValue::getInitialAmount() {
    return species()->getInitialAmount();
}

double MxSpeciesValue::getInitialConcentration() {
    return species()->getInitialConcentration();
}

bool MxSpeciesValue::getConstant() {
    return species()->getConstant();
}

double MxSpeciesValue::secrete(const double &amount, struct MxParticleList *to, double *distance) {
    if(to) return MxSecreteUptake::secrete(this, amount, *to);
    else if(distance) return MxSecreteUptake::secrete(this, amount, *distance);

    mx_error(E_FAIL, "No particle list or distance specified");
    return 0.0;
}
