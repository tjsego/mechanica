/*
 * MxStyle.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXSTYLE_HPP_
#define SRC_RENDERING_MXSTYLE_HPP_

#include <MxStyle.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

#include "../io/mx_io.h"

#include <string>

typedef Magnum::Color4 (*ColorMapperFunc)(struct MxColorMapper *mapper, struct MxParticle *p);

/**
 * @brief The Mechanica style type
 */
struct CAPI_EXPORT MxStyle
{
    Magnum::Color3 color;
    uint32_t flags;
    
    /**
     * @brief Color mapper of this style
     */
    struct MxColorMapper *mapper = NULL;
    
    ColorMapperFunc mapper_func;

    MxStyle(const Magnum::Color3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);

    /**
     * @brief Construct a new style
     * 
     * @param color name of color
     * @param visible visibility flag
     * @param flags style flags
     * @param cmap color mapper
     */
    MxStyle(const std::string &color, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);
    MxStyle(const MxStyle &other);

    int init(const Magnum::Color3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);

    /**
     * @brief Set the color by name
     * 
     * @param colorName name of color
     * @return HRESULT 
     */
    HRESULT setColor(const std::string &colorName);
    HRESULT setFlag(StyleFlags flag, bool value);
    
    Magnum::Color4 map_color(struct MxParticle *p);

    const bool getVisible() const;
    void setVisible(const bool &visible);
    MxColorMapper *getColorMap() const;
    void setColorMap(const std::string &colorMap);
    void setColorMapper(MxColorMapper *cmap);

    /**
     * @brief Construct and apply a new color map for a particle type and species
     * 
     * @param partType particle type
     * @param speciesName name of species
     * @param name name of color map
     * @param min minimum value of map
     * @param max maximum value of map
     */
    void newColorMapper(struct MxParticleType *partType,
                        const std::string &speciesName, 
                        const std::string &name="rainbow", 
                        float min=0.0f, float max=1.0f);

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
     * @return MxStyle* 
     */
    static MxStyle *fromString(const std::string &str);
};

namespace mx { namespace io {

template <>
HRESULT toFile(const MxStyle &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxStyle *dataElement);

}};

#endif /* SRC_RENDERING_MXSTYLE_HPP_ */
