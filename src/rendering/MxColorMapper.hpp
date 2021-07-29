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


struct MxColorMapper
{
    ColorMapperFunc map;
    int species_index;
    
    float min_val;
    float max_val;

    MxColorMapper() {}
    /**
     * Makes a new color map.
     * the first arg, args should be a MxParticleType object.
     *
     * since this is a style, presently this method will not set any error
     * conditions, but will set a warnign, and return null on failure.
     */
    MxColorMapper(struct MxParticleType *partType,
                  const std::string &speciesName, 
                  const std::string &name="rainbow", float min=0.0f, float max=1.0f);
    ~MxColorMapper() {};
    
    /**
     * tries to set the colormap , if the map doesn't exist,
     * does not do anything and return false.
     */
    bool set_colormap(const std::string& s);

    static std::vector<std::string> getNames();
};

#endif /* SRC_RENDERING_MXCOLORMAPPER_H_ */
