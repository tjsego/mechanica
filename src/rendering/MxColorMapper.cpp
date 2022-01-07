/*
 * MxColorMapper.cpp
 *
 *  Created on: Dec 27, 2020
 *      Author: andy
 */

#include <rendering/MxColorMapper.hpp>
#include "MxParticle.h"
#include <MxLogger.h>
#include <state/MxSpeciesList.h>
#include <mx_error.h>
#include <MxSimulator.h>

#include "colormaps/colormaps.h"

struct ColormapItem {
    const char* name;
    ColorMapperFunc func;
};


#define COLORMAP_FUNCTION(CMAP) \
static Magnum::Color4 CMAP (MxColorMapper *cm, struct MxParticle *part) { \
    float s = part->state_vector->fvec[cm->species_index];                \
    return Magnum::Color4{colormaps::all:: CMAP (s), 1};                  \
}\


COLORMAP_FUNCTION(grey_0_100_c0);
COLORMAP_FUNCTION(grey_10_95_c0);
COLORMAP_FUNCTION(kryw_0_100_c71);
COLORMAP_FUNCTION(kryw_0_97_c73);
COLORMAP_FUNCTION(green_5_95_c69);
COLORMAP_FUNCTION(blue_5_95_c73);
COLORMAP_FUNCTION(bmw_5_95_c86);
COLORMAP_FUNCTION(bmy_10_95_c71);
COLORMAP_FUNCTION(bgyw_15_100_c67);
COLORMAP_FUNCTION(gow_60_85_c27);
COLORMAP_FUNCTION(gow_65_90_c35);
COLORMAP_FUNCTION(blue_95_50_c20);
COLORMAP_FUNCTION(red_0_50_c52);
COLORMAP_FUNCTION(green_0_46_c42);
COLORMAP_FUNCTION(blue_0_44_c57);
COLORMAP_FUNCTION(bwr_40_95_c42);
COLORMAP_FUNCTION(gwv_55_95_c39);
COLORMAP_FUNCTION(gwr_55_95_c38);
COLORMAP_FUNCTION(bkr_55_10_c35);
COLORMAP_FUNCTION(bky_60_10_c30);
COLORMAP_FUNCTION(bjy_30_90_c45);
COLORMAP_FUNCTION(bjr_30_55_c53);
COLORMAP_FUNCTION(bwr_55_98_c37);
COLORMAP_FUNCTION(cwm_80_100_c22);
COLORMAP_FUNCTION(bgymr_45_85_c67);
COLORMAP_FUNCTION(bgyrm_35_85_c69);
COLORMAP_FUNCTION(bgyr_35_85_c72);
COLORMAP_FUNCTION(mrybm_35_75_c68);
COLORMAP_FUNCTION(mygbm_30_95_c78);
COLORMAP_FUNCTION(wrwbw_40_90_c42);
COLORMAP_FUNCTION(grey_15_85_c0);
COLORMAP_FUNCTION(cgo_70_c39);
COLORMAP_FUNCTION(cgo_80_c38);
COLORMAP_FUNCTION(cm_70_c39);
COLORMAP_FUNCTION(cjo_70_c25);
COLORMAP_FUNCTION(cjm_75_c23);
COLORMAP_FUNCTION(kbjyw_5_95_c25);
COLORMAP_FUNCTION(kbw_5_98_c40);
COLORMAP_FUNCTION(bwy_60_95_c32);
COLORMAP_FUNCTION(bwyk_16_96_c31);
COLORMAP_FUNCTION(wywb_55_96_c33);
COLORMAP_FUNCTION(krjcw_5_98_c46);
COLORMAP_FUNCTION(krjcw_5_95_c24);
COLORMAP_FUNCTION(cwr_75_98_c20);
COLORMAP_FUNCTION(cwrk_40_100_c20);
COLORMAP_FUNCTION(wrwc_70_100_c20);

ColormapItem colormap_items[] = {
    {"Gray", grey_0_100_c0},
    {"DarkGray", grey_10_95_c0},
    {"Heat", kryw_0_100_c71},
    {"DarkHeat", kryw_0_97_c73},
    {"Green", green_5_95_c69},
    {"Blue", blue_5_95_c73},
    {"BlueMagentaWhite", bmw_5_95_c86},
    {"BlueMagentaYellow", bmy_10_95_c71},
    {"BGYW", bgyw_15_100_c67},
    {"GreenOrangeWhite", gow_60_85_c27},
    {"DarkGreenOrangeWhite", gow_65_90_c35},
    {"LightBlue", blue_95_50_c20},
    {"Red", red_0_50_c52},
    {"DarkGreen", green_0_46_c42},
    {"DarkBlue", blue_0_44_c57},
    {"BlueWhiteRed", bwr_40_95_c42},
    {"GreenWhiteViolet", gwv_55_95_c39},
    {"GreenWhiteRed", gwr_55_95_c38},
    {"BlueBlackRed", bkr_55_10_c35},
    {"BlueBlackYellow", bky_60_10_c30},
    {"BlueGrayYellow", bjy_30_90_c45},
    {"BlueGrayRed", bjr_30_55_c53},
    {"BluwWhiteRed", bwr_55_98_c37},
    {"CyanWhiteMagenta", cwm_80_100_c22},
    {"BGYMR", bgymr_45_85_c67},
    {"DarkBGYMR", bgyrm_35_85_c69},
    {"Rainbow", bgyr_35_85_c72},
    {"CyclicMRYBM", mrybm_35_75_c68},
    {"CyclicMYGBM", mygbm_30_95_c78},
    {"CyclicWRWBW", wrwbw_40_90_c42},
    {"CyclicGray", grey_15_85_c0},
    {"DarkCyanGreenOrange", cgo_70_c39},
    {"CyanGreenOrange", cgo_80_c38},
    {"CyanMagenta", cm_70_c39},
    {"CyanGrayOrange", cjo_70_c25},
    {"CyanGrayMagenta", cjm_75_c23},
    {"KBJW", kbjyw_5_95_c25},
    {"BlackBlueWhite", kbw_5_98_c40},
    {"BlueWhiteYellow", bwy_60_95_c32},
    {"CyclicBWYK", bwyk_16_96_c31},
    {"CyclicWYWB", wywb_55_96_c33},
    {"KRJCW", krjcw_5_98_c46},
    {"DarkKRJCW", krjcw_5_95_c24},
    {"CyclicCyanWhiteRed", cwr_75_98_c20},
    {"CyclicCWRK", cwrk_40_100_c20},
    {"CyclicWRWC", wrwc_70_100_c20}
};

static bool iequals(const std::string& a, const std::string& b)
{
    unsigned int sz = a.size();
    if (b.size() != sz)
        return false;
    for (unsigned int i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

static int colormap_index_of_name(const char* s) {
    const int size = sizeof(colormap_items) / sizeof(ColormapItem);
    
    for(int i = 0; i < size; ++i) {
        if (iequals(s, colormap_items[i].name)) {
            return i;
        }
    }
    return -1;
}

bool MxColorMapper::set_colormap(const std::string& s) {
    int index = colormap_index_of_name(s.c_str());
    
    if(index >= 0) {
        this->map = colormap_items[index].func;
        
        MxSimulator::get()->redraw();
        
        return true;
    }
    return false;
}

MxColorMapper::MxColorMapper(MxParticleType *partType,
                             const std::string &speciesName,
                             const std::string &name, float min, float max) {
    
    if(partType->species == NULL) {
        std::string msg = "can not create color map for particle type \"";
        msg += partType->name;
        msg += "\" without any species defined";
        mx_exp(std::invalid_argument(msg));
        return;
    }
    
    int index = partType->species->index_of(speciesName);
    Log(LOG_DEBUG) << "Got species index: " << index;
    
    if(index < 0) {
        std::string msg = "can not create color map for particle type \"";
        msg += partType->name;
        msg += "\", does not contain species \"";
        msg += speciesName;
        msg += "\"";
        mx_exp(std::invalid_argument(msg));
        return;
    }
    
    int cmap_index = colormap_index_of_name(name.c_str());
    
    if(cmap_index >= 0) {
        this->map = colormap_items[cmap_index].func;
    }
    else {
        this->map = bgyr_35_85_c72;
    }
    
    this->species_index = index;
    this->min_val = min;
    this->max_val = max;
}

std::vector<std::string> MxColorMapper::getNames() {
    std::vector<std::string> result;
    for (auto c : colormap_items) result.push_back(c.name);
    return result;
}


namespace mx { namespace io {

template <>
HRESULT toFile(const MxColorMapper &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    fe = new MxIOElement();
    if(toFile(dataElement.species_index, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["species_index"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.min_val, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["min_val"] = fe;

    fe = new MxIOElement();
    if(toFile(dataElement.max_val, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["max_val"] = fe;
    
    const int numMaps = sizeof(colormap_items) / sizeof(ColormapItem);
    
    std::string cMapName = "";
    for(unsigned int i = 0; i < numMaps; i++) {
        auto cMap = colormap_items[i];
        if(dataElement.map == cMap.func) {
            cMapName = std::string(cMap.name);
            break;
        }
    }

    if(cMapName.size() > 0) {
        fe = new MxIOElement();
        if(toFile(cMapName, metaData, fe) != S_OK) 
            return E_FAIL;
        fe->parent = fileElement;
        fileElement->children["colorMap"] = fe;
    }

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxColorMapper *dataElement) {

    MxIOChildMap::const_iterator feItr;

    feItr = fileElement.children.find("species_index");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &dataElement->species_index) != S_OK) 
        return E_FAIL;
    
    feItr = fileElement.children.find("min_val");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &dataElement->min_val) != S_OK) 
        return E_FAIL;

    feItr = fileElement.children.find("max_val");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &dataElement->max_val) != S_OK) 
        return E_FAIL;

    feItr = fileElement.children.find("colorMap");
    if(feItr != fileElement.children.end()) {
        std::string cMapName;
        if(fromFile(*feItr->second, metaData, &cMapName) != S_OK) 
            return E_FAIL;

        auto idx = colormap_index_of_name(cMapName.c_str());
        if(idx >= 0) 
            dataElement->map = colormap_items[idx].func;
    }

    return S_OK;
}

}};
