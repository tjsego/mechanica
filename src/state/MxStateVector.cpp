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
#include <io/MxFIO.h>

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

MxStateVector::MxStateVector(MxSpeciesList *_species, 
                             void *_owner, 
                             MxStateVector *existingStateVector, 
                             uint32_t flags, 
                             void *_data) 
{
    Log(LOG_DEBUG) << "Creating state vector";

    this->species = _species;
    if(_owner) this->owner = _owner;
    
    this->size = _species->size();
    
    const int fvec_offset = 0;
    const int fvec_size = this->size * sizeof(float);
    const int q_offset = fvec_offset + fvec_size;
    const int q_size = this->size * sizeof(float);
    const int flags_offset = q_offset + q_size;
    const int flags_size = this->size * sizeof(int32_t);
    
    if(!_data) {
        this->flags |= STATEVECTOR_OWNMEMORY;
        this->data = malloc(fvec_size + q_size + flags_size);
        bzero(this->data, fvec_size + q_size + flags_size);
        this->fvec =          (float*)   ((uint8_t*)this->data + fvec_offset);
        this->q =             (float*)   ((uint8_t*)this->data + q_offset);
        this->species_flags = (uint32_t*)((uint8_t*)this->data + flags_offset);
    }

    // Copy from other state if provided; otherwise initialize from any available initial conditions
    if(existingStateVector) statevector_copy_values(this, existingStateVector);
    else {
        for(int i = 0; i < _species->size(); ++i) {
            auto _s = _species->item(i);
            if(_s->isSetInitialConcentration()) 
                this->fvec[i] = (float)_s->getInitialConcentration();
        }
    }
    
    for(int i = 0; i < _species->size(); ++i) {
        this->species_flags[i] = _species->item(i)->flags();
    }
}

MxStateVector::MxStateVector(const MxStateVector &other) : 
    MxStateVector(other.species, other.owner, const_cast<MxStateVector*>(&other), other.flags, 0)
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

std::string MxStateVector::toString() {
    return mx::io::toString(*this);
}

MxStateVector *MxStateVector::fromString(const std::string &str) {
    return mx::io::fromString<MxStateVector*>(str);
}


namespace mx { namespace io { 

template <>
HRESULT toFile(const MxStateVector &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    MxIOElement *fe;

    fe = new MxIOElement();
    fe->parent = fileElement;
    toFile(dataElement.flags, metaData, fe);
    fileElement->children["flags"] = fe;

    fe = new MxIOElement();
    fe->parent = fileElement;
    toFile(dataElement.size, metaData, fe);
    fileElement->children["size"] = fe;

    if(dataElement.species != NULL) {
        fe = new MxIOElement();
        fe->parent = fileElement;
        toFile(*dataElement.species, metaData, fe);
        fileElement->children["species"] = fe;
    }

    if(dataElement.size > 0) {
        std::vector<float> fvec, q;
        std::vector<uint32_t> species_flags;
        for(unsigned int i = 0; i < dataElement.size; i++) {
            fvec.push_back(dataElement.fvec[i]);
            q.push_back(dataElement.q[i]);
            species_flags.push_back(dataElement.species_flags[i]);
        }
        
        fe = new MxIOElement();
        fe->parent = fileElement;
        if(toFile(fvec, metaData, fe) != S_OK) 
            return E_FAIL;
        fileElement->children["quantities"] = fe;

        fe = new MxIOElement();
        fe->parent = fileElement;
        if(toFile(q, metaData, fe) != S_OK) 
            return E_FAIL;
        fileElement->children["fluxes"] = fe;

        fe = new MxIOElement();
        fe->parent = fileElement;
        if(toFile(species_flags, metaData, fe) != S_OK) 
            return E_FAIL;
        fileElement->children["species_flags"] = fe;
    }

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxStateVector **dataElement) {
    std::unordered_map<std::string, MxIOElement *>::const_iterator feItr;
    auto c = fileElement.children;

    uint32_t flags;
    feItr = c.find("flags");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &flags) != S_OK) 
        return E_FAIL;
    
    uint32_t size;
    feItr = c.find("size");
    if(feItr == c.end()) 
        return E_FAIL;
    if(fromFile(*feItr->second, metaData, &size) != S_OK) 
        return E_FAIL;

    MxSpeciesList *species = new MxSpeciesList();
    std::vector<float> fvec, q;
    std::vector<uint32_t> species_flags;

    if(size > 0) {
        feItr = c.find("species");
        if(feItr == c.end()) 
            return E_FAIL;
        fromFile(*feItr->second, metaData, species);

        feItr = c.find("quantities");
        if(feItr == c.end()) 
            return E_FAIL;
        fromFile(*feItr->second, metaData, &fvec);

        feItr = c.find("fluxes");
        if(feItr == c.end()) 
            return E_FAIL;
        fromFile(*feItr->second, metaData, &q);

        feItr = c.find("species_flags");
        if(feItr == c.end()) 
            return E_FAIL;
        fromFile(*feItr->second, metaData, &species_flags);
    }

    *dataElement = new MxStateVector(species, 0, 0, flags);

    if(size > 0) {
        for(unsigned int i = 0; i < size; i++) {

            (*dataElement)->fvec[i] = fvec[i];
            (*dataElement)->q[i] = q[i];
            (*dataElement)->species_flags[i] = species_flags[i];

        }
    }

    return S_OK;
}

}};
