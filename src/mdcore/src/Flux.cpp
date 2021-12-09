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

static void integrate_statevector(MxStateVector *s, float dt=-1.0) {
    if(dt < 0) dt = _Engine.dt;

    for(int i = 0; i < s->size; ++i) {
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
