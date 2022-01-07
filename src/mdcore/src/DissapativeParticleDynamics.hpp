/*
 * DissapativeParticleDynamics.hpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#pragma once

#include "MxPotential.h"

#ifndef _DISSAPATIVEPARTICLEDYNAMICS_HPP_
#define _DISSAPATIVEPARTICLEDYNAMICS_HPP_


struct DPDPotential : public MxPotential {
    
    /** strength of conserative interaction */
    float alpha;
    
    /** strength of dissapative interaction */
    float gamma;
    
    /** strength of random interaction */
    float sigma;

    DPDPotential(float alpha, float gamma, float sigma, float cutoff, bool shifted);

    /**
     * @brief Convert basic potential to DPD. 
     * 
     * If the basic potential is not DPD, then NULL is returned. 
     * 
     * @param pot 
     * @return DPDPotential* 
     */
    static DPDPotential *fromPot(MxPotential *pot);

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation
     * 
     * @param str 
     * @return MxPotential* 
     */
    static DPDPotential *fromString(const std::string &str);
};


namespace mx { namespace io {

HRESULT toFile(DPDPotential *dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, DPDPotential **dataElement);

}};

DPDPotential *DPDPotential_fromStr(const std::string &str);

#endif /* _DISSAPATIVEPARTICLEDYNAMICS_HPP_ */
