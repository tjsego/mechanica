/*
 * dpd_eval.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#pragma once
#ifndef SRC_MDCORE_SRC_DPD_EVAL_HPP_
#define SRC_MDCORE_SRC_DPD_EVAL_HPP_

#include "MxPotential.h"
#include "DissapativeParticleDynamics.hpp"
#include "smoothing_kernel.hpp"
#include <random>




MX_ALWAYS_INLINE bool dpd_eval(DPDPotential *p, float gaussian,
                               MxParticle *pi, MxParticle *pj, float* dx, float r2 , FPTYPE *energy) {
    
    static const float delta = 1.f / std::sqrt(_Engine.dt);
    static const float epsilon = std::numeric_limits<float>::epsilon();
    
    float r = std::sqrt(r2);
    float ro = r < epsilon ? epsilon : r;

    r = p->flags & POTENTIAL_SHIFTED ? r - (pi->radius + pj->radius) : r;

    if(r > p->b) {
        return false;
    }
    r = r >= p->a ? r : p->a;
    
    // unit vector
    MxVector3f e = {dx[0] / ro, dx[1] / ro, dx[2] / ro};
    
    MxVector3f v = pi->velocity - pj->velocity;
    
    // conservative force
    float omega_c = r < 0.f ?  1.f : (1 - r / p->b);
    
    float fc = p->alpha * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = -p->gamma * omega_d * e.dot(v);
    
    float fr = p->sigma * omega_c * delta;
    
    float f = fc + fd + fr;
    
    pj->force = {pj->f[0] - f * e[0], pj->f[1] - f * e[1], pj->f[2] - f * e[2] };
    
    pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
    
    // TODO: correct energy
    *energy = 0;
    
    return true;
}

MX_ALWAYS_INLINE bool dpd_boundary_eval(DPDPotential *p, float gaussian,
                               MxParticle *pi, float &rj, const float *velocity, const float* dx, float r2 , FPTYPE *energy) {
    
    static const float delta = 1.f / std::sqrt(_Engine.dt);
    static const float epsilon = std::numeric_limits<float>::epsilon();
    
    float r = std::sqrt(r2);
    float ro = r < epsilon ? epsilon : r;

    r = p->flags & POTENTIAL_SHIFTED ? r - (pi->radius + rj) : r;

    if(r > p->b) {
        return false;
    }
    r = r >= p->a ? r : p->a;
    
    // unit vector
    MxVector3f e = {dx[0] / ro, dx[1] / ro, dx[2] / ro};
    
    MxVector3f v = {pi->velocity[0] - velocity[0], pi->velocity[1] - velocity[1], pi->velocity[2] - velocity[2]};
    
    // conservative force
    float omega_c = r < 0.f ?  1.f : (1 - r / p->b);
    
    float fc = p->alpha * omega_c;
    
    // dissapative force
    float omega_d = omega_c * omega_c;
    
    float fd = -p->gamma * omega_d * e.dot(v);
    
    float fr = p->sigma * omega_c * delta;
    
    float f = fc + fd + fr;
    
    pi->force = {pi->f[0] + f * e[0], pi->f[1] + f * e[1], pi->f[2] + f * e[2] };
    
    // TODO: correct energy
    *energy = 0;
    
    return true;
}

// MX_ALWAYS_INLINE void potential_eval ( struct MxPotential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f ) {


#endif /* SRC_MDCORE_SRC_DPD_EVAL_HPP_ */
