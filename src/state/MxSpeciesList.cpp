/**
 * @file MxSpeciesList.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines container for MxSpecies; derived from carbon CSpeciesList.cpp written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#include "MxSpeciesList.h"

#include <MxLogger.h>

// #include <sbml/Species.h>
#include <sstream>
#include <iostream>
#include <iterator>


int32_t MxSpeciesList::index_of(const std::string &s) 
{
    int32_t result = -1;
    
    auto i = species_map.find(s);
    
    if(i != species_map.end()) {
        result = std::distance(species_map.begin(), i);
    }
    
    return result;
}

int32_t MxSpeciesList::size() 
{
    return species_map.size();
}

MxSpecies* MxSpeciesList::item(const std::string &s) 
{
    auto i = species_map.find(s);
    if(i != species_map.end()) {
        return i->second;
    }
    return NULL;
}

MxSpecies* MxSpeciesList::item(int32_t index) 
{
    if(index < species_map.size()) {
        auto i = species_map.begin();
        i = std::next(i, index);
        return i->second;
    }
    return NULL;
}

HRESULT MxSpeciesList::insert(MxSpecies* s)
{
    Log(LOG_DEBUG) << "Inserting species: " << s->getId();
    
    species_map.emplace(s->getId(), s);

    Log(LOG_DEBUG) << size();
    Log(LOG_DEBUG) << str();
    return S_OK;
}

HRESULT MxSpeciesList::insert(const std::string &s) {
    return insert(new MxSpecies(s));
}

std::string MxSpeciesList::str() {
    std::stringstream  ss;
    
    ss << "SpeciesList([";
    for(int i = 0; i < size(); ++i) {
        MxSpecies *s = item(i);
        ss << "'" << s->getId() << "'";

        if(i+1 < size()) {
            ss << ", ";
        }
    }
    ss << "])";
    return ss.str();
}

MxSpeciesList::~MxSpeciesList() {
    
    for (auto &i : species_map) {
        delete i.second;
    }
    species_map.clear();

}