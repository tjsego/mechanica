/*
 * MxBind.hpp
 *
 *  Created on: Feb 13, 2021
 *      Author: andy
 */

#pragma once
#ifndef SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_
#define SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_

#include "MxBoundaryConditions.hpp"
#include "MxForce.h"
#include "MxPotential.h"
#include <bond.h>

#include <utility>

// TODO: document interface of MxBind


struct CAPI_EXPORT MxBind {

    static HRESULT particles(MxPotential *p, MxParticle *a, MxParticle *b);

    /**
     * @brief Bind a potential to a pair of particle types. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param a The first type
     * @param b The second type
     * @param bound Flag signifying whether this potential exclusively operates on particles of different clusters, optional
     * @return HRESULT 
     */
    static HRESULT types(MxPotential *p, MxParticleType *a, MxParticleType *b, bool bound=false);

    static HRESULT cuboid(MxPotential *p, MxParticleType *t);

    static HRESULT boundaryConditions(MxPotential *p, MxBoundaryConditions *bcs, MxParticleType *t);

    static HRESULT boundaryCondition(MxPotential *p, MxBoundaryCondition *bc, MxParticleType *t);

    static HRESULT force(MxForce *force, MxParticleType *a_type, const std::string* coupling_symbol=NULL);

    static HRESULT bonds(MxPotential* potential,
                         MxParticleList *particles, 
                         const double &cutoff, 
                         std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs=NULL, 
                         const double &half_life=std::numeric_limits<double>::max(), 
                         const double &bond_energy=std::numeric_limits<double>::max(), 
                         uint32_t flags=0, 
                         std::vector<MxBondHandle*> **out=NULL);

    static HRESULT sphere(MxPotential *potential,
                          const int &n,
                          MxVector3f *center=NULL,
                          const float &radius=1.0,
                          std::pair<float, float> *phi=NULL, 
                          MxParticleType *type=NULL, 
                          std::pair<MxParticleList*, std::vector<MxBondHandle*>*> **out=NULL);

    #ifdef SWIGPYTHON
    static std::vector<MxBondHandle*> _bondsPy(MxPotential* potential,
                                               MxParticleList *particles, 
                                               const double &cutoff, 
                                               std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs=NULL, 
                                               double *half_life=NULL, 
                                               double *bond_energy=NULL, 
                                               uint32_t flags=0) 
    {
        auto _half_life = half_life ? *half_life : std::numeric_limits<double>::max();
        auto _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<double>::max();

        static std::vector<MxBondHandle*> *result = NULL;
        MxBind::bonds(potential, particles, cutoff, pairs, _half_life, _bond_energy, flags, &result);
        return *result;
    }

    static std::pair<MxParticleList*, std::vector<MxBondHandle*>*> _spherePy(MxPotential *potential, 
                                                                             const int &n, 
                                                                             MxVector3f *center=NULL, 
                                                                             const float &radius=1.0, 
                                                                             float *phi0=NULL, 
                                                                             float *phi1=NULL, 
                                                                             MxParticleType *type=NULL) 
    {
        std::pair<float, float> *phi;
        if(phi0 && phi1) phi = new std::pair<float, float>(*phi0, *phi1);
        else phi = NULL;

        std::pair<MxParticleList*, std::vector<MxBondHandle*>*> *result = NULL;
        MxBind::sphere(potential, n, center,radius, phi, type, &result);
        return *result;
    }
    #endif

};

#endif /* SRC_MDCORE_SRC_MXPOTENTIALBIND_HPP_ */
