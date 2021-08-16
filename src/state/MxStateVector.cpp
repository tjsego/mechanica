/**
 * @file MxStateVector.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the state vector; derived from carbon CStateVector.cpp written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#include <state/MxStateVector.h>
#include "MxSpeciesList.h"
#include <MxLogger.h>
#include <mx_error.h>

#include <sbml/Species.h>

#include <iostream>
#include <sstream>

// reset the species values based on the values specified in the species.
void MxStateVector::reset() {
    for(int i = 0; i < species->size(); ++i) {
        MxSpecies *s = species->item(i);
        float value = 0.f;
        if(s->isSetInitialConcentration()) {
            value = s->getInitialConcentration();
        }
        fvec[i] = value;
    }
}

static void statevector_copy_values(MxStateVector *newVec, const MxStateVector* oldVec) {
    for(int i = 0; i < oldVec->species->size(); ++i) {
        MxSpecies *species = oldVec->species->item(i);
        
        int j = newVec->species->index_of(species->getId().c_str());
        if(j >= 0) {
            newVec->fvec[j] = oldVec->fvec[i];
        }
    }
}

const std::string MxStateVector::str() const {
    std::stringstream  ss;
    
    ss << "StateVector([";
    for(int i = 0; i < size; ++i) {
        MxSpecies *s = species->item(i);
        ss << s->species->getId();
        ss << ":";
        ss << fvec[i];
        if(i+1 < size) {
            ss << ", ";
        }
    }
    ss << "])";
    return ss.str();
}

float *MxStateVector::item(const int &i) {
    
    if(i >= 0 && i < size) return &fvec[i];
    else {
        mx_exp(std::runtime_error("state vector index out of range"));
        return NULL;
    }
}

void MxStateVector::setItem(const int &i, const float &val) {
    if(i >= 0 && i < size) fvec[i] = val;
    else {
        mx_exp(std::runtime_error("state vector index out of range"));
    }
}

MxStateVector::MxStateVector() : species(new MxSpeciesList()) {}

MxStateVector::MxStateVector(MxSpeciesList *species, 
                             void *owner, 
                             MxStateVector *existingStateVector, 
                             uint32_t flags, 
                             void *data) 
{
    Log(LOG_DEBUG) << "Creating state vector";

    this->species = species;
    if(owner) this->owner = owner;
    
    this->size = species->size();
    
    const int fvec_offset = 0;
    const int fvec_size = this->size * sizeof(float);
    const int q_offset = fvec_offset + fvec_size;
    const int q_size = this->size * sizeof(float);
    const int flags_offset = q_offset + q_size;
    const int flags_size = this->size * sizeof(int32_t);
    
    if(!data) {
        this->flags |= STATEVECTOR_OWNMEMORY;
        this->data = malloc(fvec_size + q_size + flags_size);
        bzero(this->data, fvec_size + q_size + flags_size);
        this->fvec =          (float*)   ((uint8_t*)this->data + fvec_offset);
        this->q =             (float*)   ((uint8_t*)this->data + q_offset);
        this->species_flags = (uint32_t*)((uint8_t*)this->data + flags_offset);
    }

    if(existingStateVector) statevector_copy_values(this, existingStateVector);
    
    for(int i = 0; i < species->size(); ++i) {
        this->species_flags[i] = species->item(i)->flags();
    }
}

MxStateVector::MxStateVector(const MxStateVector &other) : 
    MxStateVector(other.species, other.owner, const_cast<MxStateVector*>(&other), other.flags, other.data)
{}

MxStateVector::~MxStateVector() {
    if(!owner) {
        delete species;
        species = 0;
    }

    if(flags & STATEVECTOR_OWNMEMORY) {
        free(fvec);
    }
}
