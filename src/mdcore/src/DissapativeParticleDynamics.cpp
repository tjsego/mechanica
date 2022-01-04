/*
 * DissipativeParticleDynamics.cpp
 *
 *  Created on: Feb 7, 2021
 *      Author: andy
 */

#include <DissapativeParticleDynamics.hpp>

#include <cmath>
#include <limits>


#define DPD_SELF(handle) DPDPotential *self = ((DPDPotential*)(handle))


DPDPotential::DPDPotential(float alpha, float gamma, float sigma, float cutoff, bool shifted) : MxPotential() {
    this->kind = POTENTIAL_KIND_DPD;
    this->alpha = alpha;
    this->gamma = gamma;
    this->sigma = sigma;
    this->a = std::sqrt(std::numeric_limits<float>::epsilon());
    this->b = cutoff;
    this->name = "Dissapative Particle Dynamics";
    if(shifted) {
        this->flags |= POTENTIAL_SHIFTED;
    }
}


namespace mx { namespace io {

#define MXDPDPOTENTIALIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXDPDPOTENTIALIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

HRESULT toFile(DPDPotential *dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXDPDPOTENTIALIOTOEASY(fe, "kind", dataElement->kind);
    MXDPDPOTENTIALIOTOEASY(fe, "alpha", dataElement->alpha);
    MXDPDPOTENTIALIOTOEASY(fe, "gamma", dataElement->gamma);
    MXDPDPOTENTIALIOTOEASY(fe, "sigma", dataElement->sigma);
    MXDPDPOTENTIALIOTOEASY(fe, "a", dataElement->a);
    MXDPDPOTENTIALIOTOEASY(fe, "b", dataElement->b);
    MXDPDPOTENTIALIOTOEASY(fe, "name", std::string(dataElement->name));
    MXDPDPOTENTIALIOTOEASY(fe, "flags", dataElement->flags);

    fileElement->type = "DPDPotential";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, DPDPotential **dataElement) {

    MxIOChildMap::const_iterator feItr;

    uint32_t kind, flags;
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "kind", &kind);
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "flags", &flags);

    float alpha, gamma, sigma, b;
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "alpha", &alpha);
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "gamma", &gamma);
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "sigma", &sigma);
    MXDPDPOTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "b", &b);

    *dataElement = new DPDPotential(alpha, gamma, sigma, b, flags & POTENTIAL_SHIFTED);

    return S_OK;
}

}};
