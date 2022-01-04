/*
 * NOMStyle.cpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#include <rendering/NOMStyle.hpp>
#include <engine.h>
#include <space.h>
#include <MxUtil.h>
#include <mx_error.h>
#include "MxColorMapper.hpp"


HRESULT NOMStyle::setColor(const std::string &colorName) {
    color = Color3_Parse(colorName);
    return S_OK;
}

HRESULT NOMStyle::setFlag(StyleFlags flag, bool value) {
    if(flag == STYLE_VISIBLE) {
        if(value) this->flags |= STYLE_VISIBLE;
        else this->flags &= ~STYLE_VISIBLE;
        return space_update_style(&_Engine.s);
    }
    return mx_error(E_FAIL, "invalid flag id");
}

Magnum::Color4 NOMStyle::map_color(struct MxParticle *p) {
    if(mapper_func) {
        return mapper_func(mapper, p);
    }
    return Magnum::Color4{color, 1};
};

NOMStyle::NOMStyle(const Magnum::Color3 *color, const bool &visible, uint32_t flags, MxColorMapper *cmap) : mapper_func(NULL) {
    init(color, visible, flags, cmap);
}

NOMStyle::NOMStyle(const std::string &color, const bool &visible, uint32_t flags, MxColorMapper *cmap) : 
    NOMStyle()
{
    auto c = Color3_Parse(color);
    init(&c, visible, flags, cmap);
}

NOMStyle::NOMStyle(const NOMStyle &other) {
    init(&other.color, true, other.flags, other.mapper);
}

const bool NOMStyle::getVisible() const {
    return flags & STYLE_VISIBLE;
}

void NOMStyle::setVisible(const bool &visible) {
    setFlag(STYLE_VISIBLE, visible);
}

MxColorMapper *NOMStyle::getColorMap() const {
    return mapper;
}

void NOMStyle::setColorMap(const std::string &colorMap) {
    try {
        mapper->set_colormap(colorMap);
        mapper_func = mapper->map;
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }
}

void NOMStyle::setColorMapper(MxColorMapper *cmap) {
    if(cmap) {
        this->mapper = cmap;
        this->mapper_func = this->mapper->map;
    }
}

void NOMStyle::newColorMapper(struct MxParticleType *partType,
                              const std::string &speciesName, 
                              const std::string &name, 
                              float min, float max) 
{
    setColorMapper(new MxColorMapper(partType, speciesName, name, min, max));
}

int NOMStyle::init(const Magnum::Color3 *color, const bool &visible, uint32_t flags, MxColorMapper *cmap) {
    this->flags = flags;

    this->color = color ? *color : Color3_Parse("steelblue");

    setVisible(visible);
    setColorMapper(cmap);

    return S_OK;
}


namespace mx { namespace io {

template <>
HRESULT toFile(const NOMStyle &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 
    
    MxIOElement *fe;

    fe = new MxIOElement();
    MxVector3f color = {dataElement.color.r(), dataElement.color.g(), dataElement.color.b()};
    if(toFile(color, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["color"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.flags, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["flags"] = fe;

    if(dataElement.mapper != NULL) {
        fe = new MxIOElement();
        if(toFile(*dataElement.mapper, metaData, fe) != S_OK) 
            return E_FAIL;
        fe->parent = fileElement;
        fileElement->children["mapper"] = fe;
    }

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, NOMStyle *dataElement) { 

    MxIOChildMap::const_iterator feItr;

    MxVector3f color;
    feItr = fileElement.children.find("color");
    if(feItr != fileElement.children.end() && fromFile(*feItr->second, metaData, &color) != S_OK)
        return E_FAIL;
    dataElement->color = {color.x(), color.y(), color.z()};

    feItr = fileElement.children.find("flags");
    if(feItr != fileElement.children.end() && fromFile(*feItr->second, metaData, &dataElement->flags) != S_OK)
        return E_FAIL;

    feItr = fileElement.children.find("mapper");
    if(feItr != fileElement.children.end()) {
        MxColorMapper *mapper = new MxColorMapper();
        if(fromFile(*feItr->second, metaData, mapper) != S_OK) 
            return E_FAIL;
        dataElement->setColorMapper(mapper);
    }

    return S_OK;
}

}};
