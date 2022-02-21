/*
 * MxSecreteUptake.cpp
 *
 *  Created on: Jan 6, 2021
 *      Author: andy
 */

#include <MxSecreteUptake.hpp>
#include <MxParticle.h>
#include <../../mx_error.h>
#include <../../state/MxSpeciesList.h>
#include <../../state/MxStateVector.h>
#include <MxParticleList.hpp>
#include <metrics.h>
#include <engine.h>
#include <iostream>





HRESULT MxSecrete_AmountToParticles(struct MxSpeciesValue *species,
        double amount, uint16_t nr_parts,
        int32_t *parts, double *secreted)
{
    MxStateVector *stateVector = species->state_vector;
    MxSpecies *s = stateVector->species->item(species->index);
    const std::string& speciesName = s->getId();
    
    float amountToRemove = amount < stateVector->fvec[species->index] ? amount : stateVector->fvec[species->index];
    
    struct ParticleId {
        MxParticle *part;
        int32_t index;
    };
    
    std::vector<ParticleId> pvec;
    
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        
        int index;
        
        if(p && p->state_vector && (index = p->state_vector->species->index_of(speciesName.c_str())) >= 0) {
            pvec.push_back({p, index});
        }
    }
    
    if(pvec.size() > 0) {
        float amountPer = amountToRemove / pvec.size();
        for(ParticleId& p : pvec) {
            p.part->state_vector->fvec[p.index] += amountPer;
        }
        stateVector->fvec[species->index] -= amountToRemove;
        if(secreted) {
            *secreted = amountToRemove;
        }
    }
    
    return S_OK;
}

HRESULT MxSecrete_AmountWithinDistance(struct MxSpeciesValue *species,
        double amount, double radius,
        const std::set<short int> *typeIds, double *secreted)
{
    MxParticle *part = (MxParticle*)species->state_vector->owner;
    uint16_t nr_parts = 0;
    int32_t *parts = NULL;
    
    MxParticle_Neighbors(part, radius, typeIds, &nr_parts, &parts);
    
    return MxSecrete_AmountToParticles(species, amount, nr_parts, parts, secreted);
}


double MxSecreteUptake::secrete(MxSpeciesValue *species, const double &amount, const MxParticleList &to) {
    double secreted = 0;
    try{
        if(FAILED(MxSecrete_AmountToParticles(species, amount, to.nr_parts, to.parts, &secreted))) return NULL;
    }
    catch(const std::exception &e) {
        
    }
    return secreted;
}

double MxSecreteUptake::secrete(MxSpeciesValue *species, const double &amount, const double &distance) {
    double secreted = 0;
    
    MxParticle *part = (MxParticle*)species->state_vector->owner;
    if(!part) {
        mx_exp(std::runtime_error("species state vector has no owner"));
        return 0.0;
    }
    
    try{
        // take into account the radius of this particle.
        double radius = (double)part->radius + distance;
        std::set<short int> ids = (std::set<short int>)MxParticleType::particleTypeIds();
        if(FAILED(MxSecrete_AmountWithinDistance(species, amount, radius, &ids, &secreted))) return NULL;
    }
    catch(const std::exception &e) {
        
    }
    
    return secreted;
}
