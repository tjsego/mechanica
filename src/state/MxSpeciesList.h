/**
 * @file MxSpeciesList.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines container for MxSpecies; derived from carbon CSpeciesList.hpp written by Andy Somogyi
 * @date 2021-07-03
 * 
 */

#ifndef SRC_STATE_MXSPECIESLIST_H_
#define SRC_STATE_MXSPECIESLIST_H_

#include "MxSpecies.h"
#include <mx_port.h>
#include <string>
#include <map>

struct MxSpeciesList
{
    // get the index of a species name, arg should be
    // a py_unicode object.
    // returns >= 0 on success, -1 on failure,
    // does not set any error.
    int32_t index_of(const std::string &s);
    
    int32_t size();
    
    MxSpecies *item(int32_t index);
    
    /**
     * @brief Get a species by name
     * 
     * @param s name of species
     * @return MxSpecies* 
     */
    MxSpecies *item(const std::string &s);
    
    /**
     * @brief Inserts a species
     * 
     * @return HRESULT 
     */
    HRESULT insert(MxSpecies*);
    HRESULT insert(const std::string &s);

    std::string str();

    MxSpeciesList() {};

    ~MxSpeciesList();
    
private:
    
    typedef std::map<std::string, MxSpecies*> Map;
    Map species_map;
};

#endif /* SRC_STATE_MXSPECIESLIST_H_ */
