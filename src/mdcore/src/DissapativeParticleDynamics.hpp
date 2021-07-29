/*
 * DissapativeParticleDynamics.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#pragma once

#include "MxPotential.h"

#ifndef _DISSAPATIVEPARTICLEDYNAMICS_HPP_
#define _DISSAPATIVEPARTICLEDYNAMICS_HPP_


struct DPDPotential : public MxPotential {
    
    // strength of conserative interaction
    float alpha;
    
    // strength of dissapative interaction
    float gamma;
    
    // strength of random interaction
    float sigma;

    DPDPotential(float alpha, float gamma, float sigma, float cutoff, bool shifted);
};

#endif /* _DISSAPATIVEPARTICLEDYNAMICS_HPP_ */
