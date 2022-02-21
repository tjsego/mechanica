/*
 * Flux.cpp
 *
 *  Created on: Dec 21, 2020
 *      Author: andy
 */

#include <Flux.hpp>
#include "MxParticle.h"
#include "../../state/MxSpeciesList.h"
#include "../../state/MxStateVector.h"
#include "flux_eval.hpp"
#include "space.h"
#include "engine.h"
#include "../../MxLogger.h"
#include "../../mx_error.h"
#include "../../MxUtil.h"
#include <../../io/MxFIO.h>

MxFluxes *MxFluxes::create(FluxKind kind, MxParticleType *a, MxParticleType *b,
                           const std::string& name, float k, float decay, float target) 
{
    
    if(!a || !b) {
        throw std::invalid_argument("Invalid particle types");
    }
    
    if(!a->species) {
        std::string msg = std::string("particle type ") + a->name + " does not have any defined species";
        throw std::invalid_argument(msg);
    }
    
    if(!b->species) {
        std::string msg = std::string("particle type ") + b->name + " does not have any defined species";
        throw std::invalid_argument(msg);
    }
    
    int index_a = a->species->index_of(name.c_str());
    int index_b = b->species->index_of(name.c_str());
    
    if(index_a < 0) {
        std::string msg = std::string("particle type ") +
        a->name + " does not have species " + name;
        throw std::invalid_argument(msg);
    }
    
    if(index_b < 0) {
        std::string msg = std::string("particle type ") +
        b->name + " does not have species " + name;
        throw std::invalid_argument(msg);
    }
    
    MxFluxes *fluxes = engine_getfluxes(&_Engine, a->id, b->id);
    
    if(fluxes == NULL) {
        fluxes = MxFluxes::newFluxes(8);
    }
    
    fluxes = MxFluxes::addFlux(kind, fluxes, a->id, b->id, index_a, index_b, k, decay, target);
    
    engine_addfluxes(&_Engine, fluxes, a->id, b->id);
    
    return fluxes;
}

MxFluxes *MxFluxes::fluxFick(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &decay) {
    try {
        return MxFluxes::create(FLUX_FICK, A, B, name, k, decay, 0.f);
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxFluxes *MxFluxes::flux(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &decay) {
    return fluxFick(A, B, name, k, decay);
}

MxFluxes *MxFluxes::secrete(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &target, const float &decay) {
    try {
        return MxFluxes::create(FLUX_SECRETE, A, B, name, k, decay, target);
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxFluxes *MxFluxes::uptake(MxParticleType *A, MxParticleType *B, const std::string &name, const float &k, const float &target, const float &decay) {
    try {
        return MxFluxes::create(FLUX_UPTAKE, A, B, name, k, decay, target);
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

std::string MxFluxes::toString() {
    
    // todo: fix type deduction for mx::io::toString<MxFluxes>

    MxIOElement *fe = new MxIOElement();
    MxMetaData metaData;
    if(mx::io::toFile(*this, metaData, fe) != S_OK) 
        return "";
    return mx::io::toStr(fe, metaData);
}

MxFluxes *MxFluxes::fromString(const std::string &str) {
    return new MxFluxes(mx::io::fromString<MxFluxes>(str));
}

static void integrate_statevector(MxStateVector *s, float dt=-1.0) {
    if(dt < 0) dt = _Engine.dt;

    for(int i = 0; i < s->size; ++i) {
        s->species_flags[i] = (uint32_t)s->species->item(i)->flags();
        float konst = (s->species_flags[i] & SPECIES_KONSTANT) ? 0.f : 1.f;
        s->fvec[i] += dt * s->q[i] * konst;
        s->q[i] = 0; // clear flux for next step
    }
}

HRESULT MxFluxes_integrate(space_cell *c, float dt) {
    MxParticle *p;
    MxStateVector *s;
    
    for(int i = 0; i < c->count; ++i) {
        p = &c->parts[i];
        s = p->state_vector;
        
        if(s) {
            integrate_statevector(s, dt);
        }
    }
    
    return S_OK;
}

HRESULT MxFluxes_integrate(int cellId) {
    return MxFluxes_integrate(&_Engine.s.cells[cellId]);
}

MxFluxes *MxFluxes::addFlux(FluxKind kind, MxFluxes *fluxes,
                            int16_t typeId_a, int16_t typeId_b,
                            int32_t index_a, int32_t index_b,
                            float k, float decay, float target) {
    Log(LOG_TRACE);

    int i = 0;
    if(fluxes->size + 1 < fluxes->fluxes_size * MX_SIMD_SIZE) {
        i = fluxes->fluxes[0].size;
        fluxes->size += 1;
        fluxes->fluxes[0].size += 1;
    }
    else {
        std::string msg = "currently only ";
        msg += std::to_string(MX_SIMD_SIZE) + " flux species supported, please let the Mechanica development team know you want more. ";
        throw std::logic_error(msg);
    }
    
    MxFlux *flux = &fluxes->fluxes[0];
    
    flux->kinds[i] = kind;
    flux->type_ids[i].a = typeId_a;
    flux->type_ids[i].b = typeId_b;
    flux->indices_a[i] = index_a;
    flux->indices_b[i] = index_b;
    flux->coef[i] = k;
    flux->decay_coef[i] = decay;
    flux->target[i] = target;
    
    return fluxes;
}

MxFluxes* MxFluxes::newFluxes(int32_t init_size) {
    Log(LOG_TRACE);

    struct MxFluxes *obj = NULL;
    
    int32_t blocks = std::ceil((double)init_size / MX_SIMD_SIZE);
    
    int total_size = sizeof(MxFluxes) + blocks * sizeof(MxFlux);

    /* allocate the potential */
    if ((obj = (MxFluxes * )MxAligned_Malloc(total_size, 16 )) == NULL ) {
        return NULL;
    }
    
    ::memset(obj, 0, total_size);
    
    obj->size = 0;
    obj->fluxes_size = blocks;

    return obj;
}


namespace mx { namespace io {

#define MXFLUXIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXFLUXIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const TypeIdPair &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFLUXIOTOEASY(fe, "a", dataElement.a);
    MXFLUXIOTOEASY(fe, "b", dataElement.b);

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, TypeIdPair *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "a", &dataElement->a);
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "b", &dataElement->b);

    return S_OK;
}

template <>
HRESULT toFile(const MxFlux &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXFLUXIOTOEASY(fe, "size", dataElement.size);
    std::vector<int8_t> kinds;
    std::vector<TypeIdPair> type_ids;
    std::vector<int32_t> indices_a;
    std::vector<int32_t> indices_b;
    std::vector<float> coef;
    std::vector<float> decay_coef;
    std::vector<float> target;

    for(unsigned int i = 0; i < MX_SIMD_SIZE; i++) {
        kinds.push_back(dataElement.kinds[i]);
        type_ids.push_back(dataElement.type_ids[i]);
        indices_a.push_back(dataElement.indices_a[i]);
        indices_b.push_back(dataElement.indices_b[i]);
        coef.push_back(dataElement.coef[i]);
        decay_coef.push_back(dataElement.decay_coef[i]);
        target.push_back(dataElement.target[i]);
    }

    MXFLUXIOTOEASY(fe, "kinds", kinds);
    MXFLUXIOTOEASY(fe, "type_ids", type_ids);
    MXFLUXIOTOEASY(fe, "indices_a", indices_a);
    MXFLUXIOTOEASY(fe, "indices_b", indices_b);
    MXFLUXIOTOEASY(fe, "coef", coef);
    MXFLUXIOTOEASY(fe, "decay_coef", decay_coef);
    MXFLUXIOTOEASY(fe, "target", target);

    fileElement->type = "Flux";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxFlux *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "size", &dataElement->size);
    
    std::vector<int8_t> kinds;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "kinds", &kinds);

    std::vector<TypeIdPair> type_ids;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "type_ids", &type_ids);
    
    std::vector<int32_t> indices_a;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "indices_a", &indices_a);
    
    std::vector<int32_t> indices_b;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "indices_b", &indices_b);
    
    std::vector<float> coef;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "coef", &coef);
    
    std::vector<float> decay_coef;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "decay_coef", &decay_coef);
    
    std::vector<float> target;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "target", &target);

    for(unsigned int i = 0; i < MX_SIMD_SIZE; i++) {
        
        dataElement->kinds[i] = kinds[i];
        dataElement->type_ids[i] = type_ids[i];
        dataElement->indices_a[i] = indices_a[i];
        dataElement->indices_b[i] = indices_b[i];
        dataElement->coef[i] = coef[i];
        dataElement->decay_coef[i] = decay_coef[i];
        dataElement->target[i] = target[i];

    }
    
    return S_OK;
}

template <>
HRESULT toFile(const MxFluxes &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFLUXIOTOEASY(fe, "fluxes_size", dataElement.fluxes_size);
    
    std::vector<MxFlux> fluxes;
    for(unsigned int i = 0; i < dataElement.size; i++) 
        fluxes.push_back(dataElement.fluxes[i]);
    MXFLUXIOTOEASY(fe, "fluxes", fluxes);

    fileElement->type = "Fluxes";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxFluxes *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "fluxes_size", &dataElement->fluxes_size);
    
    std::vector<MxFlux> fluxes;
    MXFLUXIOFROMEASY(feItr, fileElement.children, metaData, "fluxes", &fluxes);
    MxFlux flux = fluxes[0];
    for(unsigned int i = 0; i < fluxes.size(); i++) { 
        dataElement = MxFluxes::addFlux((FluxKind)flux.kinds[i], 
                                        dataElement, 
                                        flux.type_ids[i].a, flux.type_ids[i].b, 
                                        flux.indices_a[i], flux.indices_b[i], 
                                        flux.coef[i], flux.decay_coef[i], flux.target[i]);
    }
    
    return S_OK;
}

}};
