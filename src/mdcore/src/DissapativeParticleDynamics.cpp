/*
 * DissipativeParticleDynamics.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#include <DissapativeParticleDynamics.hpp>

#include <cmath>
#include <limits>


#define DPD_SELF(handle) DPDPotential *self = ((DPDPotential*)(handle))


DPDPotential::DPDPotential(float alpha, float gamma, float sigma, float cutoff, bool shifted) : MxPotential() {
    this->kind = POTENTIAL_KIND_DPD;
    this->alpha = alpha;
    this->gamma = gamma;
    this->sigma = sigma;
    this->a = std::sqrt(std::numeric_limits<float>::epsilon());
    this->b = cutoff;
    this->name = "Dissapative Particle Dynamics";
    if(shifted) {
        this->flags |= POTENTIAL_SHIFTED;
    }
}
