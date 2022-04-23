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

struct CAPI_EXPORT MxBind {

    /**
     * @brief Bind a potential to a pair of particles
     * 
     * @param p The potential
     * @param a The first particle
     * @param b The second particle
     * @return HRESULT 
     */
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

    /**
     * @brief Bind a potential to a pair of particle type and all boundary conditions. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param t The particle type
     * @return HRESULT 
     */
    static HRESULT boundaryConditions(MxPotential *p, MxParticleType *t);
    
    /**
     * @brief Bind a potential to a pair of particle type and a boundary conditions. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param p The potential
     * @param bcs The boundary condition
     * @param t The particle type
     * @return HRESULT 
     */
    static HRESULT boundaryCondition(MxPotential *p, MxBoundaryCondition *bc, MxParticleType *t);

    /**
     * @brief Bind a force to a particle type
     * 
     * @param force The force
     * @param a_type The particle type
     * @return HRESULT 
     */
    static HRESULT force(MxForce *force, MxParticleType *a_type);

    /**
     * @brief Bind a force to a particle type with magnitude proportional to a species amount
     * 
     * @param force The force
     * @param a_type The particle type
     * @param coupling_symbol The symbol of the species
     * @return HRESULT 
     */
    static HRESULT force(MxForce *force, MxParticleType *a_type, const std::string& coupling_symbol);

    /**
     * @brief Create bonds for a set of pairs of particles
     * 
     * @param potential The bond potential
     * @param particles The list of particles
     * @param cutoff Interaction cutoff
     * @param pairs Pairs to bind
     * @param half_life Bond half life
     * @param bond_energy Bond dissociation energy
     * @param flags Bond flags
     * @param out List of created bonds
     * @return HRESULT 
     */
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
    /**
     * @brief Create bonds for a set of pairs of particles
     * 
     * @param potential The bond potential
     * @param particles The list of particles
     * @param cutoff Interaction cutoff
     * @param pairs Pairs to bind
     * @param half_life Bond half life
     * @param bond_energy Bond dissociation energy
     * @param flags Bond flags
     * @return std::vector<MxBondHandle*> 
     */
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
