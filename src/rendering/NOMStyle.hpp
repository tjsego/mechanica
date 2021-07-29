/*
 * NOMStyle.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_NOMSTYLE_HPP_
#define SRC_RENDERING_NOMSTYLE_HPP_

#include <NOMStyle.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

#include <string>

typedef Magnum::Color4 (*ColorMapperFunc)(struct MxColorMapper *mapper, struct MxParticle *p);

struct CAPI_EXPORT NOMStyle
{
    Magnum::Color3 color;
    uint32_t flags;
    
    struct MxColorMapper *mapper;
    
    ColorMapperFunc mapper_func;

    NOMStyle(const Magnum::Color3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);
    NOMStyle(const std::string &color, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);
    NOMStyle(const NOMStyle &other);

    int init(const Magnum::Color3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, MxColorMapper *cmap=NULL);

    inline HRESULT setColor(const std::string &colorName);
    HRESULT setFlag(StyleFlags flag, bool value);
    
    inline Magnum::Color4 map_color(struct MxParticle *p);

    inline const bool getVisible() const;
    inline void setVisible(const bool &visible);
    inline MxColorMapper *getColorMap() const;
    inline void setColorMap(const std::string &colorMap);
    inline void setColorMapper(MxColorMapper *cmap);
    inline void newColorMapper(struct MxParticleType *partType,
                               const std::string &speciesName, 
                               const std::string &name="rainbow", 
                               float min=0.0f, float max=1.0f);
};

#endif /* SRC_RENDERING_NOMSTYLE_HPP_ */
