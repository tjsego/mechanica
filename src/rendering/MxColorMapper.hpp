/*
 * MxColorMapper.h
 *
 *  Created on: Dec 27, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXCOLORMAPPER_H_
#define SRC_RENDERING_MXCOLORMAPPER_H_

#include <rendering/NOMStyle.hpp>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

#include <vector>

/**
 * @brief The color mapping type
 */
struct MxColorMapper
{
    ColorMapperFunc map;
    int species_index;
    
    /**
     * @brief minimum value of map
     */
    float min_val;

    /**
     * @brief maximum value of map
     */
    float max_val;

    MxColorMapper() {}
    
    /**
     * @brief Construct a new color map for a particle type and species
     * 
     * @param partType particle type
     * @param speciesName name of species
     * @param name name of color mapper function
     * @param min minimum value of map
     * @param max maximum value of map
     */
    MxColorMapper(struct MxParticleType *partType,
                  const std::string &speciesName, 
                  const std::string &name="rainbow", float min=0.0f, float max=1.0f);
    ~MxColorMapper() {};
    
    /**
     * @brief Try to set the colormap. 
     * 
     * If the map doesn't exist, does not do anything and returns false.
     * 
     * @param s name of color map
     * @return true on success
     */
    bool set_colormap(const std::string& s);

    static std::vector<std::string> getNames();
};

namespace mx { namespace io {

template <>
HRESULT toFile(const MxColorMapper &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxColorMapper *dataElement);

}};

#endif /* SRC_RENDERING_MXCOLORMAPPER_H_ */
