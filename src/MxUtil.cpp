/*
 * MxParticles.cpp
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <MxUtil.h>
#include <MxThreadPool.hpp>
#include <mx_error.h>

#include <mdcore_config.h>
#include <engine.h>

#include "Magnum/Mesh.h"
#include "Magnum/Math/Vector3.h"
#include "Magnum/MeshTools/RemoveDuplicates.h"
#include "Magnum/MeshTools/Subdivide.h"
#include "Magnum/Trade/ArrayAllocator.h"
#include "Magnum/Trade/MeshData.h"

#include "metrics.h"

#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <chrono>

static MxRandomType *MxRandom_p = NULL;
static unsigned int randomSeed = 0;

MxRandomType &MxRandomEngine() {
    if(MxRandom_p == NULL) {
        MxRandom_p = new MxRandomType();
        MxRandom_p->seed(randomSeed);
    }
    return *MxRandom_p;
}

unsigned int MxGetSeed() {
    return randomSeed;
}

HRESULT MxSetSeed(const unsigned int *_seed) {

    unsigned int seed;

    if(_seed == NULL) {
        srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        seed = (unsigned int)rand()*((std::numeric_limits<unsigned int>::max)()-1);
    } 
    else seed = *_seed;

    randomSeed = seed;

    MxRandomType &MxRandom = MxRandomEngine();
    MxRandom.seed(seed);
    return S_OK;
}

const char* MxColor3Names[] = {
    "AliceBlue",
    "AntiqueWhite",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "Black",
    "BlanchedAlmond",
    "Blue",
    "BlueViolet",
    "Brown",
    "BurlyWood",
    "CadetBlue",
    "Chartreuse",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkBlue",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGray",
    "DarkGreen",
    "DarkKhaki",
    "DarkMagenta",
    "DarkOliveGreen",
    "Darkorange",
    "DarkOrchid",
    "DarkRed",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkSlateBlue",
    "DarkSlateGray",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DimGray",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Gray",
    "Green",
    "GreenYellow",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Indigo",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "Maroon",
    "MediumAquaMarine",
    "MediumBlue",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MidnightBlue",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "Navy",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Salmon",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "Tan",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
    "SpaBlue",
    "Pumpkin",
    "OleumYellow",
    "SGIPurple",
    NULL
};

typedef float (*force_2body_fn)(struct EnergyMinimizer* p, MxVector3f *x1, MxVector3f *x2,
MxVector3f *f1, MxVector3f *f2);

typedef float (*force_1body_fn)(struct EnergyMinimizer* p, MxVector3f *p1,
MxVector3f *f1);

struct EnergyMinimizer {
    force_1body_fn force_1body;
    force_2body_fn force_2body;
    int max_outer_iter;
    int max_inner_iter;
    float outer_de;
    float inner_de;
    float cutoff;
};

static void energy_minimize(EnergyMinimizer *p, std::vector<MxVector3f> &points);

static float sphere_2body(EnergyMinimizer* p, MxVector3f *x1, MxVector3f *x2,
                          MxVector3f *f1, MxVector3f *f2);

static float sphere_1body(EnergyMinimizer* p, MxVector3f *p1,
                          MxVector3f *f1) ;

static MxVector3f random_point_disk(std::uniform_real_distribution<float> &uniform01) {
    auto &MxRandom = MxRandomEngine();
    float r = sqrt(uniform01(MxRandom));
    float theta = 2 * M_PI * uniform01(MxRandom);
    return MxVector3f(r * cos(theta), r * sin(theta), 0.);
}

static std::vector<MxVector3f> random_points_disk(int n) {

    std::vector<MxVector3f> result(n);

    try {
        std::uniform_real_distribution<float> uniform01(0.0, 1.0);

        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = random_point_disk(uniform01);
        }
    }
    catch (const std::exception &e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static MxVector3f random_point_sphere(std::uniform_real_distribution<float> &uniform01) {
    float radius = 1.0;

    auto &MxRandom = MxRandomEngine();
    float theta = 2 * M_PI * uniform01(MxRandom);
    float phi = acos(1 - 2 * uniform01(MxRandom));
    float x = radius * sin(phi) * cos(theta);
    float y = radius * sin(phi) * sin(theta);
    float z = radius * cos(phi);

    return MxVector3f(x, y, z);
}

static std::vector<MxVector3f> random_points_sphere(int n) {

    std::vector<MxVector3f> result(n);

    try {
        std::vector<MxVector3f> points(n);
        
        float radius = 1.0;

        std::uniform_real_distribution<float> uniform01(0.0, 1.0);

        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = random_point_sphere(uniform01);
        }
        
        EnergyMinimizer em;
        em.force_1body = sphere_1body;
        em.force_2body = sphere_2body;
        em.max_outer_iter = 10;
        em.cutoff = 0.2;
        
        energy_minimize(&em, result);

    }
    catch (const std::exception &e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static MxVector3f random_point_solidsphere(std::uniform_real_distribution<float> &uniform01) {
    auto &MxRandom = MxRandomEngine();
    float theta = 2 * M_PI * uniform01(MxRandom);
    float phi = acos(1 - 2 * uniform01(MxRandom));
    float r = std::cbrt(uniform01(MxRandom));
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);

    return MxVector3f(x, y, z);
}

static std::vector<MxVector3f> random_points_solidsphere(int n) {

    std::vector<MxVector3f> result(n);

    try {
        std::uniform_real_distribution<float> uniform01(0.0, 1.0);

        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = random_point_solidsphere(uniform01);
        }

    }
    catch (const std::exception& e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static MxVector3f random_point_solidsphere_shell(std::uniform_real_distribution<float> &uniform01, 
                                                 const float &dr, 
                                                 const float &cos0, 
                                                 const float &cos1) 
{
    auto &MxRandom = MxRandomEngine();
    float theta = 2 * M_PI * uniform01(MxRandom);
    float phi = acos(cos0 - (cos0-cos1) * uniform01(MxRandom));
    float r = std::cbrt((1-dr) + dr * uniform01(MxRandom));
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);

    return MxVector3f(x, y, z);
}

static std::vector<MxVector3f> random_points_solidsphere_shell(int n, const float &dr=1.0, const float &phi0=0, const float &phi1=M_PI) {
    std::vector<MxVector3f> result(n);

    float cos0 = std::cos(phi0);
    float cos1 = std::cos(phi1);

    try {
        std::uniform_real_distribution<float> uniform01(0.0, 1.0);

        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = random_point_solidsphere_shell(uniform01, dr, cos0, cos1);
        }
    }
    catch (const std::exception& e) {
        mx_exp(e);
        result.clear();
    }
    
    return result;
}

static MxVector3f random_point_solidcube(std::uniform_real_distribution<float> &uniform01) {
    auto &MxRandom = MxRandomEngine();
    return MxVector3f(uniform01(MxRandom), uniform01(MxRandom), uniform01(MxRandom));
}

static std::vector<MxVector3f> random_points_solidcube(int n) {
    std::vector<MxVector3f> result(n);

    try {
        std::uniform_real_distribution<float> uniform01(-0.5, 0.5);

        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = random_point_solidcube(uniform01);
        }

    }
    catch (const std::exception& e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static std::vector<MxVector3f> points_solidcube(int n) {
    std::vector<MxVector3f> result(n);
    
    try {
        if(n < 8) mx_exp(std::runtime_error("minimum 8 points in cube"));

        std::uniform_real_distribution<float> uniform01(-0.5, 0.5);
        auto &MxRandom = MxRandomEngine();
        
        for(auto p = result.begin(); p != result.end(); ++p) {
            *p = MxVector3f(uniform01(MxRandom), uniform01(MxRandom), uniform01(MxRandom));
        }
        
    }
    catch (const std::exception& e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static std::vector<MxVector3f> points_ring(int n) {
    std::vector<MxVector3f> result(n);

    try {
        float radius = 1.0;
        const float phi = M_PI / 2.;
        const float theta_i = 2 * M_PI / n;

        for(unsigned i = 0; i < result.size(); ++i) {
            float theta = i * theta_i;
            float x = radius * sin(phi) * cos(theta);
            float y = radius * sin(phi) * sin(theta);
            float z = radius * cos(phi);

            result[i] = MxVector3f(x, y, z);
        }

    }
    catch (const std::exception& e) {
        mx_exp(e);
        result.clear();
    }

    return result;
}

static std::vector<MxVector3f> points_sphere(int n) {
    std::vector<MxVector3f> result;
    std::vector<int32_t> indices;
    if(Mx_Icosphere(n, 0, M_PI, result, indices) != S_OK) result.clear();
    return result;
}

MxVector3f MxRandomPoint(const MxPointsType &kind, 
                         const float &dr, 
                         const float &phi0, 
                         const float &phi1) 
{
    try {
        std::uniform_real_distribution<float> uniform01(0.0, 1.0);

        switch(kind) {
        case MxPointsType::Sphere:
            return random_point_sphere(uniform01);
        case MxPointsType::Disk:
            return random_point_disk(uniform01);
        case MxPointsType::SolidCube:
            return random_point_solidcube(uniform01);
        case MxPointsType::SolidSphere: {
            return random_point_solidsphere_shell(uniform01, dr, std::cos(phi0), std::cos(phi1));
        }
        default:
            mx_exp(std::runtime_error("invalid kind"));
            return MxVector3f();
        }
    }
    catch (const std::exception& e) {
        mx_exp(e); 
    }

    return MxVector3f();
}

std::vector<MxVector3f> MxRandomPoints(const MxPointsType &kind, 
                                       const int &n, 
                                       const float &dr, 
                                       const float &phi0, 
                                       const float &phi1)
{
    try {
        switch(kind) {
        case MxPointsType::Sphere:
            return random_points_sphere(n);
        case MxPointsType::Disk:
            return random_points_disk(n);
        case MxPointsType::SolidCube:
            return random_points_solidcube(n);
        case MxPointsType::SolidSphere: {
            return random_points_solidsphere_shell(n, dr, phi0, phi1);
        }
        default:
            mx_exp(std::runtime_error("invalid kind"));
            return std::vector<MxVector3f>();
        }
    }
    catch (const std::exception& e) {
        mx_exp(e); 
    }

    return std::vector<MxVector3f>();
}

std::vector<MxVector3f> MxPoints(const MxPointsType &kind, const int &n)
{
    try {
        switch(kind) {
            case MxPointsType::Ring:
                return points_ring(n);
            case MxPointsType::Sphere:
                return points_sphere(n);
            default:
                mx_exp(std::runtime_error("invalid kind"));
                return std::vector<MxVector3f>();
        }
    }
    catch (const std::exception& e) {
        mx_exp(e);
    }

    return std::vector<MxVector3f>();
}

// todo: expand MxFilledCubeUniform to rectangular cuboids
std::vector<MxVector3f> MxFilledCubeUniform(const MxVector3f &corner1, 
                                            const MxVector3f &corner2, 
                                            const int &nParticlesX, 
                                            const int &nParticlesY, 
                                            const int &nParticlesZ) 
{
    if(nParticlesX < 2 || nParticlesY < 2 || nParticlesZ < 2) 
        mx_exp(std::range_error("Must have 2 or more particles in each direction"));

    std::vector<MxVector3f> result;
    MxVector3f cubeSpan = corner2 - corner1;

    for(int i = 0; i < nParticlesX; i++) {
        MxVector3f incMatX(float(i) / (float(nParticlesX) - 1.0), 0.0, 0.0);

        for(int j = 0; j < nParticlesY; j++) {
            MxVector3f incMatY(0.0, float(j) / (float(nParticlesY) - 1.0), 0.0);

            for(int k = 0; k < nParticlesZ; k++) {
                MxVector3f incMatZ(0.0, 0.0, float(k) / (float(nParticlesZ) - 1.0));

                MxMatrix3f incMat(incMatX, incMatY, incMatZ);
                result.push_back(corner1 + incMat * cubeSpan);
            }
        }
    }

    return result;
}

// todo: expand MxFilledCubeRandom to rectangular cuboids
std::vector<MxVector3f> MxFilledCubeRandom(const MxVector3f &corner1, const MxVector3f &corner2, const int &nParticles) {
    std::vector<MxVector3f> result;

    std::uniform_real_distribution<float> disx(corner1[0], corner2[0]);
    std::uniform_real_distribution<float> disy(corner1[1], corner2[1]);
    std::uniform_real_distribution<float> disz(corner1[2], corner2[2]);
    auto &MxRandom = MxRandomEngine();

    for(int i = 0; i < nParticles; ++i) {
        result.push_back(MxVector3f{disx(MxRandom), disy(MxRandom), disz(MxRandom)});

    }

    return result;
}

std::vector<std::string> MxColor3_Names() {
    return std::vector<std::string>(std::begin(MxColor3Names), std::end(MxColor3Names) - 1);
}

Magnum::Color3 Color3_Parse(const std::string &s)
{
    if(s.length() < 2) {
        // TODO ???
        return Magnum::Color3{};
    }

    // #ff6347
    if(s.length() >= 0 && s[0] == '#') {
        std::string srgb = s.substr(1, s.length() - 1);

        char* ss;
        unsigned long rgb = strtoul(srgb.c_str(), &ss, 16);

        return Magnum::Color3::fromSrgb(rgb);
    }

    std::string str = s;
    std::transform(str.begin(), str.end(),str.begin(), ::toupper);

    // TODO, thread safe???
    static std::unordered_map<std::string, Magnum::Color3> colors;
    if(colors.size() == 0) {
        colors["INDIANRED"]         = Magnum::Color3::fromSrgb(0xCD5C5C);
        colors["LIGHTCORAL"]        = Magnum::Color3::fromSrgb(0xF08080);
        colors["SALMON"]            = Magnum::Color3::fromSrgb(0xFA8072);
        colors["DARKSALMON"]        = Magnum::Color3::fromSrgb(0xE9967A);
        colors["LIGHTSALMON"]       = Magnum::Color3::fromSrgb(0xFFA07A);
        colors["CRIMSON"]           = Magnum::Color3::fromSrgb(0xDC143C);
        colors["RED"]               = Magnum::Color3::fromSrgb(0xFF0000);
        colors["FIREBRICK"]         = Magnum::Color3::fromSrgb(0xB22222);
        colors["DARKRED"]           = Magnum::Color3::fromSrgb(0x8B0000);
        colors["PINK"]              = Magnum::Color3::fromSrgb(0xFFC0CB);
        colors["LIGHTPINK"]         = Magnum::Color3::fromSrgb(0xFFB6C1);
        colors["HOTPINK"]           = Magnum::Color3::fromSrgb(0xFF69B4);
        colors["DEEPPINK"]          = Magnum::Color3::fromSrgb(0xFF1493);
        colors["MEDIUMVIOLETRED"]   = Magnum::Color3::fromSrgb(0xC71585);
        colors["PALEVIOLETRED"]     = Magnum::Color3::fromSrgb(0xDB7093);
        colors["LIGHTSALMON"]       = Magnum::Color3::fromSrgb(0xFFA07A);
        colors["CORAL"]             = Magnum::Color3::fromSrgb(0xFF7F50);
        colors["TOMATO"]            = Magnum::Color3::fromSrgb(0xFF6347);
        colors["ORANGERED"]         = Magnum::Color3::fromSrgb(0xFF4500);
        colors["DARKORANGE"]        = Magnum::Color3::fromSrgb(0xFF8C00);
        colors["ORANGE"]            = Magnum::Color3::fromSrgb(0xFFA500);
        colors["GOLD"]              = Magnum::Color3::fromSrgb(0xFFD700);
        colors["YELLOW"]            = Magnum::Color3::fromSrgb(0xFFFF00);
        colors["LIGHTYELLOW"]       = Magnum::Color3::fromSrgb(0xFFFFE0);
        colors["LEMONCHIFFON"]      = Magnum::Color3::fromSrgb(0xFFFACD);
        colors["LIGHTGOLDENRODYELLOW"] = Magnum::Color3::fromSrgb(0xFAFAD2);
        colors["PAPAYAWHIP"]        = Magnum::Color3::fromSrgb(0xFFEFD5);
        colors["MOCCASIN"]          = Magnum::Color3::fromSrgb(0xFFE4B5);
        colors["PEACHPUFF"]         = Magnum::Color3::fromSrgb(0xFFDAB9);
        colors["PALEGOLDENROD"]     = Magnum::Color3::fromSrgb(0xEEE8AA);
        colors["KHAKI"]             = Magnum::Color3::fromSrgb(0xF0E68C);
        colors["DARKKHAKI"]         = Magnum::Color3::fromSrgb(0xBDB76B);
        colors["LAVENDER"]          = Magnum::Color3::fromSrgb(0xE6E6FA);
        colors["THISTLE"]           = Magnum::Color3::fromSrgb(0xD8BFD8);
        colors["PLUM"]              = Magnum::Color3::fromSrgb(0xDDA0DD);
        colors["VIOLET"]            = Magnum::Color3::fromSrgb(0xEE82EE);
        colors["ORCHID"]            = Magnum::Color3::fromSrgb(0xDA70D6);
        colors["FUCHSIA"]           = Magnum::Color3::fromSrgb(0xFF00FF);
        colors["MAGENTA"]           = Magnum::Color3::fromSrgb(0xFF00FF);
        colors["MEDIUMORCHID"]      = Magnum::Color3::fromSrgb(0xBA55D3);
        colors["MEDIUMPURPLE"]      = Magnum::Color3::fromSrgb(0x9370DB);
        colors["REBECCAPURPLE"]     = Magnum::Color3::fromSrgb(0x663399);
        colors["BLUEVIOLET"]        = Magnum::Color3::fromSrgb(0x8A2BE2);
        colors["DARKVIOLET"]        = Magnum::Color3::fromSrgb(0x9400D3);
        colors["DARKORCHID"]        = Magnum::Color3::fromSrgb(0x9932CC);
        colors["DARKMAGENTA"]       = Magnum::Color3::fromSrgb(0x8B008B);
        colors["PURPLE"]            = Magnum::Color3::fromSrgb(0x800080);
        colors["INDIGO"]            = Magnum::Color3::fromSrgb(0x4B0082);
        colors["SLATEBLUE"]         = Magnum::Color3::fromSrgb(0x6A5ACD);
        colors["DARKSLATEBLUE"]     = Magnum::Color3::fromSrgb(0x483D8B);
        colors["MEDIUMSLATEBLUE"]   = Magnum::Color3::fromSrgb(0x7B68EE);
        colors["GREENYELLOW"]       = Magnum::Color3::fromSrgb(0xADFF2F);
        colors["CHARTREUSE"]        = Magnum::Color3::fromSrgb(0x7FFF00);
        colors["LAWNGREEN"]         = Magnum::Color3::fromSrgb(0x7CFC00);
        colors["LIME"]              = Magnum::Color3::fromSrgb(0x00FF00);
        colors["LIMEGREEN"]         = Magnum::Color3::fromSrgb(0x32CD32);
        colors["PALEGREEN"]         = Magnum::Color3::fromSrgb(0x98FB98);
        colors["LIGHTGREEN"]        = Magnum::Color3::fromSrgb(0x90EE90);
        colors["MEDIUMSPRINGGREEN"] = Magnum::Color3::fromSrgb(0x00FA9A);
        colors["SPRINGGREEN"]       = Magnum::Color3::fromSrgb(0x00FF7F);
        colors["MEDIUMSEAGREEN"]    = Magnum::Color3::fromSrgb(0x3CB371);
        colors["SEAGREEN"]          = Magnum::Color3::fromSrgb(0x2E8B57);
        colors["FORESTGREEN"]       = Magnum::Color3::fromSrgb(0x228B22);
        colors["GREEN"]             = Magnum::Color3::fromSrgb(0x008000);
        colors["DARKGREEN"]         = Magnum::Color3::fromSrgb(0x006400);
        colors["YELLOWGREEN"]       = Magnum::Color3::fromSrgb(0x9ACD32);
        colors["OLIVEDRAB"]         = Magnum::Color3::fromSrgb(0x6B8E23);
        colors["OLIVE"]             = Magnum::Color3::fromSrgb(0x808000);
        colors["DARKOLIVEGREEN"]    = Magnum::Color3::fromSrgb(0x556B2F);
        colors["MEDIUMAQUAMARINE"]  = Magnum::Color3::fromSrgb(0x66CDAA);
        colors["DARKSEAGREEN"]      = Magnum::Color3::fromSrgb(0x8FBC8B);
        colors["LIGHTSEAGREEN"]     = Magnum::Color3::fromSrgb(0x20B2AA);
        colors["DARKCYAN"]          = Magnum::Color3::fromSrgb(0x008B8B);
        colors["TEAL"]              = Magnum::Color3::fromSrgb(0x008080);
        colors["AQUA"]              = Magnum::Color3::fromSrgb(0x00FFFF);
        colors["CYAN"]              = Magnum::Color3::fromSrgb(0x00FFFF);
        colors["LIGHTCYAN"]         = Magnum::Color3::fromSrgb(0xE0FFFF);
        colors["PALETURQUOISE"]     = Magnum::Color3::fromSrgb(0xAFEEEE);
        colors["AQUAMARINE"]        = Magnum::Color3::fromSrgb(0x7FFFD4);
        colors["TURQUOISE"]         = Magnum::Color3::fromSrgb(0x40E0D0);
        colors["MEDIUMTURQUOISE"]   = Magnum::Color3::fromSrgb(0x48D1CC);
        colors["DARKTURQUOISE"]     = Magnum::Color3::fromSrgb(0x00CED1);
        colors["CADETBLUE"]         = Magnum::Color3::fromSrgb(0x5F9EA0);
        colors["STEELBLUE"]         = Magnum::Color3::fromSrgb(0x4682B4);
        colors["LIGHTSTEELBLUE"]    = Magnum::Color3::fromSrgb(0xB0C4DE);
        colors["POWDERBLUE"]        = Magnum::Color3::fromSrgb(0xB0E0E6);
        colors["LIGHTBLUE"]         = Magnum::Color3::fromSrgb(0xADD8E6);
        colors["SKYBLUE"]           = Magnum::Color3::fromSrgb(0x87CEEB);
        colors["LIGHTSKYBLUE"]      = Magnum::Color3::fromSrgb(0x87CEFA);
        colors["DEEPSKYBLUE"]       = Magnum::Color3::fromSrgb(0x00BFFF);
        colors["DODGERBLUE"]        = Magnum::Color3::fromSrgb(0x1E90FF);
        colors["CORNFLOWERBLUE"]    = Magnum::Color3::fromSrgb(0x6495ED);
        colors["MEDIUMSLATEBLUE"]   = Magnum::Color3::fromSrgb(0x7B68EE);
        colors["ROYALBLUE"]         = Magnum::Color3::fromSrgb(0x4169E1);
        colors["BLUE"]              = Magnum::Color3::fromSrgb(0x0000FF);
        colors["MEDIUMBLUE"]        = Magnum::Color3::fromSrgb(0x0000CD);
        colors["DARKBLUE"]          = Magnum::Color3::fromSrgb(0x00008B);
        colors["NAVY"]              = Magnum::Color3::fromSrgb(0x000080);
        colors["MIDNIGHTBLUE"]      = Magnum::Color3::fromSrgb(0x191970);
        colors["CORNSILK"]          = Magnum::Color3::fromSrgb(0xFFF8DC);
        colors["BLANCHEDALMOND"]    = Magnum::Color3::fromSrgb(0xFFEBCD);
        colors["BISQUE"]            = Magnum::Color3::fromSrgb(0xFFE4C4);
        colors["NAVAJOWHITE"]       = Magnum::Color3::fromSrgb(0xFFDEAD);
        colors["WHEAT"]             = Magnum::Color3::fromSrgb(0xF5DEB3);
        colors["BURLYWOOD"]         = Magnum::Color3::fromSrgb(0xDEB887);
        colors["TAN"]               = Magnum::Color3::fromSrgb(0xD2B48C);
        colors["ROSYBROWN"]         = Magnum::Color3::fromSrgb(0xBC8F8F);
        colors["SANDYBROWN"]        = Magnum::Color3::fromSrgb(0xF4A460);
        colors["GOLDENROD"]         = Magnum::Color3::fromSrgb(0xDAA520);
        colors["DARKGOLDENROD"]     = Magnum::Color3::fromSrgb(0xB8860B);
        colors["PERU"]              = Magnum::Color3::fromSrgb(0xCD853F);
        colors["CHOCOLATE"]         = Magnum::Color3::fromSrgb(0xD2691E);
        colors["SADDLEBROWN"]       = Magnum::Color3::fromSrgb(0x8B4513);
        colors["SIENNA"]            = Magnum::Color3::fromSrgb(0xA0522D);
        colors["BROWN"]             = Magnum::Color3::fromSrgb(0xA52A2A);
        colors["MAROON"]            = Magnum::Color3::fromSrgb(0x800000);
        colors["WHITE"]             = Magnum::Color3::fromSrgb(0xFFFFFF);
        colors["SNOW"]              = Magnum::Color3::fromSrgb(0xFFFAFA);
        colors["HONEYDEW"]          = Magnum::Color3::fromSrgb(0xF0FFF0);
        colors["MINTCREAM"]         = Magnum::Color3::fromSrgb(0xF5FFFA);
        colors["AZURE"]             = Magnum::Color3::fromSrgb(0xF0FFFF);
        colors["ALICEBLUE"]         = Magnum::Color3::fromSrgb(0xF0F8FF);
        colors["GHOSTWHITE"]        = Magnum::Color3::fromSrgb(0xF8F8FF);
        colors["WHITESMOKE"]        = Magnum::Color3::fromSrgb(0xF5F5F5);
        colors["SEASHELL"]          = Magnum::Color3::fromSrgb(0xFFF5EE);
        colors["BEIGE"]             = Magnum::Color3::fromSrgb(0xF5F5DC);
        colors["OLDLACE"]           = Magnum::Color3::fromSrgb(0xFDF5E6);
        colors["FLORALWHITE"]       = Magnum::Color3::fromSrgb(0xFFFAF0);
        colors["IVORY"]             = Magnum::Color3::fromSrgb(0xFFFFF0);
        colors["ANTIQUEWHITE"]      = Magnum::Color3::fromSrgb(0xFAEBD7);
        colors["LINEN"]             = Magnum::Color3::fromSrgb(0xFAF0E6);
        colors["LAVENDERBLUSH"]     = Magnum::Color3::fromSrgb(0xFFF0F5);
        colors["MISTYROSE"]         = Magnum::Color3::fromSrgb(0xFFE4E1);
        colors["GAINSBORO"]         = Magnum::Color3::fromSrgb(0xDCDCDC);
        colors["LIGHTGRAY"]         = Magnum::Color3::fromSrgb(0xD3D3D3);
        colors["SILVER"]            = Magnum::Color3::fromSrgb(0xC0C0C0);
        colors["DARKGRAY"]          = Magnum::Color3::fromSrgb(0xA9A9A9);
        colors["GRAY"]              = Magnum::Color3::fromSrgb(0x808080);
        colors["DIMGRAY"]           = Magnum::Color3::fromSrgb(0x696969);
        colors["LIGHTSLATEGRAY"]    = Magnum::Color3::fromSrgb(0x778899);
        colors["SLATEGRAY"]         = Magnum::Color3::fromSrgb(0x708090);
        colors["DARKSLATEGRAY"]     = Magnum::Color3::fromSrgb(0x2F4F4F);
        colors["BLACK"]             = Magnum::Color3::fromSrgb(0x000000);
	colors["SPABLUE"]           = Magnum::Color3::fromSrgb(0x6D99D3); // Rust Oleum Spa Blue
	colors["PUMPKIN"]           = Magnum::Color3::fromSrgb(0xF65917); // Rust Oleum Pumpkin
	colors["OLEUMYELLOW"]       = Magnum::Color3::fromSrgb(0xF9CB20); // rust oleum yellow
	colors["SGIPURPLE"]         = Magnum::Color3::fromSrgb(0x6353BB); // SGI purple
    }

    std::unordered_map<std::string, Magnum::Color3>::const_iterator got =
            colors.find (str);

    if (got != colors.end()) {
        return got->second;
    }
    
    std::string warning = std::string("Warning, \"") + s + "\" is not a valid color name.";
    
    PyErr_WarnEx(PyExc_Warning, warning.c_str(), 0);

    return Magnum::Color3{};
}

constexpr uint32_t Indices[]{
    1, 2, 6,
    1, 7, 2,
    3, 4, 5,
    4, 3, 8,
    6, 5, 11,
    
    5, 6, 10,
    9, 10, 2,
    10, 9, 3,
    7, 8, 9,
    8, 7, 0,
    
    11, 0, 1,
    0, 11, 4,
    6, 2, 10,
    1, 6, 11,
    3, 5, 10,
    
    5, 4, 11,
    2, 7, 9,
    7, 1, 0,
    3, 9, 8,
    4, 8, 0
};

/* Can't be just an array of Vector3 because MSVC 2015 is special. See
 Crosshair.cpp for details. */
constexpr struct VertexSolidStrip {
    MxVector3f position;
} Vertices[] {
    {{0.0f, -0.525731f, 0.850651f}},
    {{0.850651f, 0.0f, 0.525731f}},
    {{0.850651f, 0.0f, -0.525731f}},
    {{-0.850651f, 0.0f, -0.525731f}},
    {{-0.850651f, 0.0f, 0.525731f}},
    {{-0.525731f, 0.850651f, 0.0f}},
    {{0.525731f, 0.850651f, 0.0f}},
    {{0.525731f, -0.850651f, 0.0f}},
    {{-0.525731f, -0.850651f, 0.0f}},
    {{0.0f, -0.525731f, -0.850651f}},
    {{0.0f, 0.525731f, -0.850651f}},
    {{0.0f, 0.525731f, 0.850651f}}
};

HRESULT Mx_Icosphere(const int subdivisions, float phi0, float phi1,
                    std::vector<MxVector3f> &verts,
                    std::vector<int32_t> &inds) {
    
    // TODO: sloppy ass code, needs clean up...
    // total waste computing a whole sphere and throwign large parts away.
    
    const std::size_t indexCount =
        Magnum::Containers::arraySize(Indices) * (1 << subdivisions * 2);
    
    const std::size_t vertexCount =
        Magnum::Containers::arraySize(Vertices) +
        ((indexCount - Magnum::Containers::arraySize(Indices))/3);
    
    Magnum::Containers::Array<char> indexData{indexCount*sizeof(uint32_t)};
    
    auto indices = Magnum::Containers::arrayCast<uint32_t>(indexData);
    
    std::memcpy(indices.begin(), Indices, sizeof(Indices));
    
    struct Vertex {
        MxVector3f position;
        MxVector3f normal;
    };
    
    Magnum::Containers::Array<char> vertexData;
    Magnum::Containers::arrayResize<Magnum::Trade::ArrayAllocator>(
        vertexData, Magnum::Containers::NoInit, sizeof(Vertex)*vertexCount);
    
    /* Build up the subdivided positions */
    {
        auto vertices = Magnum::Containers::arrayCast<Vertex>(vertexData);
        Magnum::Containers::StridedArrayView1D<Magnum::Vector3>
            positions{vertices, &vertices[0].position, vertices.size(), sizeof(Vertex)};
        
        for(std::size_t i = 0; i != Magnum::Containers::arraySize(Vertices); ++i)
            positions[i] = Vertices[i].position;
        
        for(std::size_t i = 0; i != subdivisions; ++i) {
            const std::size_t iterationIndexCount =
                Magnum::Containers::arraySize(Indices)*(1 << (i + 1)*2);
            
            const std::size_t iterationVertexCount =
                Magnum::Containers::arraySize(Vertices) +
                ((iterationIndexCount - Magnum::Containers::arraySize(Indices))/3);
            
            Magnum::MeshTools::subdivideInPlace(
                indices.prefix(iterationIndexCount),
                positions.prefix(iterationVertexCount),
                [](const Magnum::Vector3& a, const Magnum::Vector3& b) {
                    return (a+b).normalized();
                }
            );
        }
        
        /** @todo i need arrayShrinkAndGiveUpMemoryIfItDoesntCauseRealloc() */
        Magnum::Containers::arrayResize<Magnum::Trade::ArrayAllocator>(
            vertexData,
            Magnum::MeshTools::removeDuplicatesIndexedInPlace(
                Magnum::Containers::stridedArrayView(indices),
                Magnum::Containers::arrayCast<2, char>(positions))*sizeof(Vertex)
        );
    }
    
    /* Build up the views again with correct size, fill the normals */
    auto vertices = Magnum::Containers::arrayCast<Vertex>(vertexData);
    
    Magnum::Containers::StridedArrayView1D<Magnum::Vector3>
        positions{vertices, &vertices[0].position, vertices.size(), sizeof(Vertex)};
    
    /*
     * original code
    verts.resize(positions.size());
    inds.resize(indices.size());
    
    for(int i = 0; i < positions.size(); ++i) {
        verts[i] = positions[i];
    }
    
    for(int i = 0; i < indices.size(); ++i) {
        inds[i] = indices[i];
    }
     */
    
    // prune the top and bottom vertices 
    verts.reserve(positions.size());
    inds.reserve(indices.size());
    
    std::vector<int32_t> index_map;
    index_map.resize(positions.size());
    
    std::set<int32_t> discarded_verts;
    
    MxVector3f origin = {0.0, 0.0, 0.0};
    for(int i = 0; i < positions.size(); ++i) {
        MxVector3f position = positions[i];
        MxVector3f spherical = MxCartesianToSpherical(position, origin);
        if(spherical[2] < phi0 || spherical[2] > phi1) {
            discarded_verts.emplace(i);
            index_map[i] = -1;
        }
        else {
            index_map[i] = verts.size();
            verts.push_back(position);
        }
    }
    
    for(int i = 0; i < indices.size(); i += 3) {
        int a = indices[i];
        int b = indices[i+1];
        int c = indices[i+2];
        
        if(discarded_verts.find(a) == discarded_verts.end() &&
           discarded_verts.find(b) == discarded_verts.end() &&
           discarded_verts.find(c) == discarded_verts.end()) {
            a = index_map[a];
            b = index_map[b];
            c = index_map[c];
            assert(a >= 0 && b >= 0 && c >= 0);
            inds.push_back(a);
            inds.push_back(b);
            inds.push_back(c);
        }
    }

    return S_OK;
}

MxVector3f MxRandomVector(float mean, float std) {
    std::normal_distribution<> dist{mean,std};
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    MxRandomType &MxRandom = MxRandomEngine();
    float theta = 2 * M_PI * uniform01(MxRandom);
    float phi = acos(1 - 2 * uniform01(MxRandom));
    float r = dist(MxRandom);
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);
    return MxVector3f{x, y, z};
}

MxVector3f MxRandomUnitVector() {
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    MxRandomType &MxRandom = MxRandomEngine();
    float theta = 2 * M_PI * uniform01(MxRandom);
    float phi = acos(1 - 2 * uniform01(MxRandom));
    float r = 1.;
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);
    return MxVector3f{x, y, z};
}

static void energy_find_neighborhood(std::vector<MxVector3f> const &points,
                                     const int part,
                                     float r,
                                     std::vector<int32_t> &nbor_inds,
                                     std::vector<int32_t> &boundary_inds) {
    
    nbor_inds.resize(0);
    boundary_inds.resize(0);
  
    float r2 = r * r;
    float br2 = 4 * r * r;
    
    MxVector3f pt = points[part];
    for(int i = 0; i < points.size(); ++i) {

        MxVector3f dx = pt - points[i];
        float dx2 = dx.dot();
        if(dx2 <= r2) {
            nbor_inds.push_back(i);
        }
        if(dx2 > r2 && dx2 <= br2) {
            boundary_inds.push_back(i);
        }
    }
}

float energy_minimize_neighborhood(EnergyMinimizer *p,
                                   std::vector<int32_t> &indices,
                                   std::vector<int32_t> &boundary_indices,
                                   std::vector<MxVector3f> &points,
                                   std::vector<MxVector3f> &forces) {

    float etot = 0;

    for(int i = 0; i < 10; i++) {
    
        float e = 0;
        
        for(int j = 0; j < indices.size(); ++j) {
            forces[indices[j]] = {0.0f, 0.0f, 0.0f};
        }
        
        for(int j = 0; j < indices.size(); ++j) {
            int32_t jj = indices[j];
            // one-body force
            e += p->force_1body(p, &points[jj], &forces[jj]);
            
            // two-body force in local neighborhood
            for(int k = j+1; k < indices.size(); ++k) {
                int32_t kk = indices[k];
                e += p->force_2body(p, &points[jj], &points[kk], &forces[jj], &forces[kk]);
            }
            
            // two-body force from boundaries
            for(int k = j+1; k < boundary_indices.size(); ++k) {
                int32_t kk = boundary_indices[k];
                e += p->force_2body(p, &points[jj], &points[kk], &forces[jj], nullptr) / 2;
            }
        }
        
        for(int i = 0; i < indices.size(); ++i) {
            int32_t ii = indices[i];
            points[ii] += 1 * forces[ii];
        }
        
        etot += e;
    }
    return etot;
}

void energy_minimize(EnergyMinimizer *p, std::vector<MxVector3f> &points) {
    std::vector<int32_t> nindices(points.size()/2);
    std::vector<int32_t> bindices(points.size()/2);
    std::vector<MxVector3f> forces(points.size());
    std::uniform_int_distribution<> distrib(0, points.size() - 1);
    
    float etot[3] = {0, 0, 0};
    float de[3] = {0, 0, 0};
    int ntot = 0;
    float de_avg = 0;
    int i = 0;
    
    do {
        for(int k = 0; k < points.size(); ++k) {
            int32_t partId = k;
            energy_find_neighborhood(points, partId, p->cutoff, nindices, bindices);
            etot[i] = energy_minimize_neighborhood(p, nindices, bindices, points, forces);
        }
        i = (i + 1) % 3;
        ntot += 1;
        de[0] = etot[0] - etot[1];
        de[1] = etot[1] - etot[2];
        de[2] = etot[2] - etot[0];
        de_avg = (de[0]*de[0] + de[1]*de[1] + de[2]*de[2])/3;
        std::cout << "n:" << ntot << ", de:" << de_avg << std::endl;
    }
    while(ntot < 3 && (ntot < p->max_outer_iter));
}


float sphere_2body(EnergyMinimizer* p, MxVector3f *x1, MxVector3f *x2,
                   MxVector3f *f1, MxVector3f *f2) {
    MxVector3f dx = (*x2 - *x1); // vector from x1 -> x2
    float r = dx.length() + 0.01; // softness factor.
    if(r > p->cutoff) {
        return 0;
    }
    
    float f = 0.0001 / (r * r);
    
    *f1 = *f1 - f * dx / (2 * r);
    
    if(f2) {
        *f2 = *f2 + f * dx / (2 * r);
    }
    
    return std::abs(f);
}

float sphere_1body(EnergyMinimizer* p, MxVector3f *p1,
                   MxVector3f *f1) {
    float r = (*p1).length();

    float f = 1 * (1.0 - r); // magnitude of force.
    
    *f1 = *f1 + (f/r) * (*p1);
    
    return std::abs(f);
}

#if defined(__x86_64__) || defined(_M_X64)

// getters
std::string InstructionSet::Vendor(void)
{
    return CPU_Rep.vendor_;
}
std::string InstructionSet::Brand(void)
{
    return CPU_Rep.brand_;
}

bool InstructionSet::SSE3(void)
{
    return CPU_Rep.f_1_ECX_[0];
}
bool InstructionSet::PCLMULQDQ(void)
{
    return CPU_Rep.f_1_ECX_[1];
}
bool InstructionSet::MONITOR(void)
{
    return CPU_Rep.f_1_ECX_[3];
}
bool InstructionSet::SSSE3(void)
{
    return CPU_Rep.f_1_ECX_[9];
}
bool InstructionSet::FMA(void)
{
    return CPU_Rep.f_1_ECX_[12];
}
bool InstructionSet::CMPXCHG16B(void)
{
    return CPU_Rep.f_1_ECX_[13];
}
bool InstructionSet::SSE41(void)
{
    return CPU_Rep.f_1_ECX_[19];
}
bool InstructionSet::SSE42(void)
{
    return CPU_Rep.f_1_ECX_[20];
}
bool InstructionSet::MOVBE(void)
{
    return CPU_Rep.f_1_ECX_[22];
}
bool InstructionSet::POPCNT(void)
{
    return CPU_Rep.f_1_ECX_[23];
}
bool InstructionSet::AES(void)
{
    return CPU_Rep.f_1_ECX_[25];
}
bool InstructionSet::XSAVE(void)
{
    return CPU_Rep.f_1_ECX_[26];
}
bool InstructionSet::OSXSAVE(void)
{
    return CPU_Rep.f_1_ECX_[27];
}
bool InstructionSet::AVX(void)
{
    return CPU_Rep.f_1_ECX_[28];
}
bool InstructionSet::F16C(void)
{
    return CPU_Rep.f_1_ECX_[29];
}
bool InstructionSet::RDRAND(void)
{
    return CPU_Rep.f_1_ECX_[30];
}

bool InstructionSet::MSR(void)
{
    return CPU_Rep.f_1_EDX_[5];
}
bool InstructionSet::CX8(void)
{
    return CPU_Rep.f_1_EDX_[8];
}
bool InstructionSet::SEP(void)
{
    return CPU_Rep.f_1_EDX_[11];
}
bool InstructionSet::CMOV(void)
{
    return CPU_Rep.f_1_EDX_[15];
}
bool InstructionSet::CLFSH(void)
{
    return CPU_Rep.f_1_EDX_[19];
}
bool InstructionSet::MMX(void)
{
    return CPU_Rep.f_1_EDX_[23];
}
bool InstructionSet::FXSR(void)
{
    return CPU_Rep.f_1_EDX_[24];
}
bool InstructionSet::SSE(void)
{
    return CPU_Rep.f_1_EDX_[25];
}
bool InstructionSet::SSE2(void)
{
    return CPU_Rep.f_1_EDX_[26];
}
bool InstructionSet::FSGSBASE(void)
{
    return CPU_Rep.f_7_EBX_[0];
}
bool InstructionSet::BMI1(void)
{
    return CPU_Rep.f_7_EBX_[3];
}
bool InstructionSet::HLE(void)
{
    return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4];
}
bool InstructionSet::AVX2(void)
{
    return CPU_Rep.f_7_EBX_[5];
}
bool InstructionSet::BMI2(void)
{
    return CPU_Rep.f_7_EBX_[8];
}
bool InstructionSet::ERMS(void)
{
    return CPU_Rep.f_7_EBX_[9];
}
bool InstructionSet::INVPCID(void)
{
    return CPU_Rep.f_7_EBX_[10];
}
bool InstructionSet::RTM(void)
{
    return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11];
}
bool InstructionSet::AVX512F(void)
{
    return CPU_Rep.f_7_EBX_[16];
}
bool InstructionSet::RDSEED(void)
{
    return CPU_Rep.f_7_EBX_[18];
}
bool InstructionSet::ADX(void)
{
    return CPU_Rep.f_7_EBX_[19];
}
bool InstructionSet::AVX512PF(void)
{
    return CPU_Rep.f_7_EBX_[26];
}
bool InstructionSet::AVX512ER(void)
{
    return CPU_Rep.f_7_EBX_[27];
}
bool InstructionSet::AVX512CD(void)
{
    return CPU_Rep.f_7_EBX_[28];
}
bool InstructionSet::SHA(void)
{
    return CPU_Rep.f_7_EBX_[29];
}
bool InstructionSet::PREFETCHWT1(void)
{
    return CPU_Rep.f_7_ECX_[0];
}
bool InstructionSet::LAHF(void)
{
    return CPU_Rep.f_81_ECX_[0];
}
bool InstructionSet::LZCNT(void)
{
    return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5];
}
bool InstructionSet::ABM(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5];
}
bool InstructionSet::SSE4a(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6];
}
bool InstructionSet::XOP(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11];
}
bool InstructionSet::TBM(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21];
}
bool InstructionSet::SYSCALL(void)
{
    return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11];
}
bool InstructionSet::MMXEXT(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22];
}
bool InstructionSet::RDTSCP(void)
{
    return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27];
}
bool InstructionSet::_3DNOWEXT(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30];
}
bool InstructionSet::_3DNOW(void)
{
    return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31];
}

InstructionSet::InstructionSet() {
    featuresMap["SSE3"] = SSE3();
    featuresMap["PCLMULQDQ"] = PCLMULQDQ();
    featuresMap["MONITOR"] = MONITOR();
    featuresMap["SSSE3"] = SSSE3();
    featuresMap["FMA"] = FMA();
    featuresMap["CMPXCHG16B"] = CMPXCHG16B();
    featuresMap["SSE41"] = SSE41();
    featuresMap["SSE42"] = SSE42();
    featuresMap["MOVBE"] = MOVBE();
    featuresMap["POPCNT"] = POPCNT();
    featuresMap["AES"] = AES();
    featuresMap["XSAVE"] = XSAVE();
    featuresMap["OSXSAVE"] = OSXSAVE();
    featuresMap["AVX"] = AVX();
    featuresMap["F16C"] = F16C();
    featuresMap["RDRAND"] = RDRAND();
    featuresMap["MSR"] = MSR();
    featuresMap["CX8"] = CX8();
    featuresMap["SEP"] = SEP();
    featuresMap["CMOV"] = CMOV();
    featuresMap["CLFSH"] = CLFSH();
    featuresMap["MMX"] = MMX();
    featuresMap["FXSR"] = FXSR();
    featuresMap["SSE"] = SSE();
    featuresMap["SSE2"] = SSE2();
    featuresMap["FSGSBASE"] = FSGSBASE();
    featuresMap["BMI1"] = BMI1();
    featuresMap["HLE"] = HLE();
    featuresMap["AVX2"] = AVX2();
    featuresMap["BMI2"] = BMI2();
    featuresMap["ERMS"] = ERMS();
    featuresMap["INVPCID"] = INVPCID();
    featuresMap["RTM"] = RTM();
    featuresMap["AVX512F"] = AVX512F();
    featuresMap["RDSEED"] = RDSEED();
    featuresMap["ADX"] = ADX();
    featuresMap["AVX512PF"] = AVX512PF();
    featuresMap["AVX512ER"] = AVX512ER();
    featuresMap["AVX512CD"] = AVX512CD();
    featuresMap["SHA"] = SHA();
    featuresMap["PREFETCHWT1"] = PREFETCHWT1();
    featuresMap["LAHF"] = LAHF();
    featuresMap["LZCNT"] = LZCNT();
    featuresMap["ABM"] = ABM();
    featuresMap["SSE4a"] = SSE4a();
    featuresMap["XOP"] = XOP();
    featuresMap["TBM"] = TBM();
    featuresMap["SYSCALL"] = SYSCALL();
    featuresMap["MMXEXT"] = MMXEXT();
    featuresMap["RDTSCP"] = RDTSCP();
    featuresMap["_3DNOWEXT"] = _3DNOWEXT();
    featuresMap["_3DNOW"] = _3DNOW();
}

std::unordered_map<std::string, bool> getFeaturesMap() {
    return InstructionSet().featuresMap;
}

#else

std::unordered_map<std::string, bool> getFeaturesMap() {
    return std::unordered_map<std::string, bool>();
}

#endif

MxCompileFlags::MxCompileFlags() {
    
#ifdef _DEBUG
    flags["_DEBUG"] = 1;
#else
    flags["_DEBUG"] = 0;
#endif
    flagNames.push_back("_DEBUG");

    flags["MX_OPENMP"] = MX_OPENMP;
    flagNames.push_back("MX_OPENMP");
    flags["MX_OPENMP_BONDS"] = MX_OPENMP_BONDS;
    flagNames.push_back("MX_OPENMP_BONDS");
    flags["MX_OPENMP_INTEGRATOR"] = MX_OPENMP_INTEGRATOR;
    flagNames.push_back("MX_OPENMP_INTEGRATOR");
    flags["MX_VECTORIZE_FLUX"] = MX_VECTORIZE_FLUX;
    flagNames.push_back("MX_VECTORIZE_FLUX");
    flags["MX_VECTORIZE_FORCE"] = MX_VECTORIZE_FORCE;
    flagNames.push_back("MX_VECTORIZE_FORCE");
    flags["MX_SSE42"] = MX_SSE42;
    flagNames.push_back("MX_SSE42");
    flags["MX_AVX"] = MX_AVX;
    flagNames.push_back("MX_AVX");
    flags["MX_AVX2"] = MX_AVX2;
    flagNames.push_back("MX_AVX2");
    
#ifdef MX_THREADING
    flags["MX_THREADING"] = 1;
    flags["MX_THREADPOOL_SIZE"] = mx::ThreadPool::hardwareThreadSize();
#else
    flags["MX_THREADING"] = 0;
    flags["MX_THREADPOOL_SIZE"] = 0;
#endif
    flagNames.push_back("MX_THREADING");
    flagNames.push_back("MX_THREADPOOL_SIZE");
    
    flags["MX_SIMD_SIZE"] = MX_SIMD_SIZE;
    flagNames.push_back("MX_SIMD_SIZE");

#ifdef __SSE__
    flags["__SSE__"] = __SSE__;
#else
    flags["__SSE__"] = 0;
#endif
    flagNames.push_back("__SSE__");
    
#ifdef __SSE2__
    flags["__SSE2__"] = __SSE2__;
#else
    flags["__SSE2__"] = 0;
#endif
    flagNames.push_back("__SSE2__");
    
#ifdef __SSE3__
    flags["__SSE3__"] = __SSE3__;
#else
    flags["__SSE3__"] = 0;
#endif
    flagNames.push_back("__SSE3__");
    
#ifdef __SSE4_2__
    flags["__SSE4_2__"] = __SSE4_2__;
#else
    flags["__SSE4_2__"] = 0;
#endif
    flagNames.push_back("__SSE4_2__");
    
#ifdef __AVX__
    flags["__AVX__"] = __AVX__;
#else
    flags["__AVX__"] = 0;
#endif
    flagNames.push_back("__AVX__");
    
#ifdef __AVX2__
    flags["__AVX2__"] = __AVX2__;
#else
    flags["__AVX2__"] = 0;
#endif
    flagNames.push_back("__AVX2__");
}

const std::list<std::string> MxCompileFlags::getFlags() {
    return flagNames;
}

const int MxCompileFlags::getFlag(const std::string &_flag) {
    auto x = flags.find(_flag);
    if (x != flags.end()) return x->second;
    mx_exp(std::runtime_error("Flag not defined: " + _flag));
    return 0;
}


//  Windows
#ifdef _WIN32
#include <Windows.h>
double MxWallTime() {
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double MxCPUTime(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
        (double)(d.dwLowDateTime |
                 ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double MxWallTime() {
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * (1.0/1000000);
}
double MxCPUTime(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif




WallTime::WallTime() {
    start = MxWallTime();
}

WallTime::~WallTime() {
    _Engine.wall_time += (MxWallTime() - start);
}

PerformanceTimer::PerformanceTimer(unsigned id) {
    _start = getticks();
    _id = id;
}

PerformanceTimer::~PerformanceTimer() {
    _Engine.timers[_id] += getticks() - _start;
}

// Function that returns true if n
// is prime else returns false
static bool isPrime(uint64_t n)
{
    // Corner cases
    if (n <= 1)  return false;
    if (n <= 3)  return true;

    // This is checked so that we can skip
    // middle five numbers in below loop
    if (n%2 == 0 || n%3 == 0) return false;

    for (uint64_t i=5; i*i<=n; i=i+6)
        if (n%i == 0 || n%(i+2) == 0)
           return false;

    return true;
}

// Function to return the smallest
// prime number greater than N

uint64_t MxMath_NextPrime(const uint64_t &N)
{

    // Base case
    if (N <= 1)
        return 2;

    uint64_t prime = N;
    bool found = false;

    // Loop continuously until isPrime returns
    // true for a number greater than n
    while (!found) {
        prime++;

        if (isPrime(prime))
            found = true;
    }

    return prime;
}

std::vector<uint64_t> MxMath_FindPrimes(const uint64_t &start_prime, int n)
{
    auto result = std::vector<uint64_t>(n, 0);
    auto start_p = start_prime;
    for(auto itr = result.begin(); itr != result.end(); ++itr) {
        *itr = MxMath_NextPrime(start_p);
        start_p = *itr;
    }
    return result;
}

HRESULT MxMath_FindPrimes(uint64_t start_prime, int n, uint64_t *result)
{
    uint64_t r;
    for(unsigned int i = 0; i < n; ++i) {
        r = MxMath_NextPrime(start_prime);
        result[i] = r;
        start_prime = r;
    }
    return 0;
}

#include <Magnum/Math/Distance.h>

std::tuple<Magnum::Vector4, float> _MxTest(const MxVector3f &a, const MxVector3f &b, const MxVector3f &c) {
    try {
        Magnum::Vector4 plane = Magnum::Math::planeEquation(a, b);
        
        float dist = Magnum::Math::Distance::pointPlaneScaled(c, plane);
        
        return std::make_tuple(plane, dist);;
        
    }

    catch(const std::exception &e) {
        mx_exp(e);
    }

    return std::make_tuple(Magnum::Vector4(), 0.0f);
}

Differentiator::Differentiator(double (*f)(double), const double &xmin, const double &xmax, const double &inc_cf) :
    func(f), 
    xmin(xmin), 
    xmax(xmax), 
    inc_cf(inc_cf)
{}

double Differentiator::fnp(const double &x, const unsigned int &order) {
    if (order == 0) return this->func(x);

    double inc_0 = this->inc_cf * (this->xmax - this->xmin);
    double inc, powPreFact, powTerm, incPreFact, incTerm;

    if (x == this->xmin) {
        inc = std::min(inc_0, (this->xmax - x) / order);
        powTerm = double(order);
        powPreFact = -1.0;
        incTerm = 0.0;
        incPreFact = 1.0;
    }
    else if (x == this->xmax) {
        inc = std::min(inc_0, (x - this->xmin) / order);
        powTerm = 0.0;
        powPreFact = 1.0;
        incTerm = 0.0;
        incPreFact = -1.0;
    }
    else {
        inc = std::min(inc_0, std::min(2.0 * (this->xmax - x) / order, 2.0 * (x - this->xmin) / order));
        powTerm = 0.0;
        powPreFact = 1.0;
        incTerm = double(order) / 2.0;
        incPreFact = -1.0;
    }

    double result = 0.0;
    double xi;

    for (int i = 0; i <= order; ++i) {
        xi = x + (incTerm + incPreFact * i) * inc;
        result += Magnum::Math::binomialCoefficient(order, i) * std::pow(-1.0, powTerm + powPreFact * i) * this->func(xi);
    }

    return result / std::pow(inc, order);
}

double Differentiator::operator() (const double &x) {
    return this->fnp(x, 0);
}
