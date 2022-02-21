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
#include "../io/mx_io.h"
#include <mx_port.h>
#include <string>
#include <map>

struct MxSpeciesList
{
    /**
     * @brief Get the index of a species name
     * 
     * @param s species name
     * @return int32_t >= 0 on sucess, -1 on failure
     */
    int32_t index_of(const std::string &s);
    
    /**
     * @brief Get the size of the species
     * 
     * @return int32_t 
     */
    int32_t size();
    
    /**
     * @brief Get a species by index
     * 
     * @param index index of the species
     * @return MxSpecies* 
     */
    MxSpecies *item(int32_t index);
    
    /**
     * @brief Get a species by name
     * 
     * @param s name of species
     * @return MxSpecies* 
     */
    MxSpecies *item(const std::string &s);
    
    /**
     * @brief Insert a species
     * 
     * @return HRESULT 
     */
    HRESULT insert(MxSpecies *s);

    /**
     * @brief Insert a species by name
     * 
     * @param s name of the species
     * @return HRESULT 
     */
    HRESULT insert(const std::string &s);

    /**
     * @brief Get a string representation
     * 
     * @return std::string 
     */
    std::string str();

    MxSpeciesList() {};

    ~MxSpeciesList();

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * @param str 
     * @return MxSpeciesList* 
     */
    static MxSpeciesList *fromString(const std::string &str);
    
private:
    
    typedef std::map<std::string, MxSpecies*> Map;
    Map species_map;
};

namespace mx { namespace io { 

template <>
HRESULT toFile(const MxSpeciesList &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxSpeciesList *dataElement);

}};

#endif /* SRC_STATE_MXSPECIESLIST_H_ */
