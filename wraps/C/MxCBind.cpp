/**
 * @file MxCBind.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxBind
 * @date 2022-04-04
 */

#include "MxCBind.h"

#include "mechanica_c_private.h"

#include <mdcore/src/MxBind.hpp>
#include <MxParticle.h>
#include <MxForce.h>
#include <engine.h>


#define MXCBIND_CHECKHANDLE(h) if(!h || !h->MxObj) return E_FAIL;


//////////////////////
// Module functions //
//////////////////////


HRESULT MxCBind_particles(struct MxPotentialHandle *p, struct MxParticleHandleHandle *a, struct MxParticleHandleHandle *b) {
    MXCBIND_CHECKHANDLE(p);
    MXCBIND_CHECKHANDLE(a);
    MXCBIND_CHECKHANDLE(b);
    MxParticleHandle *ah = (MxParticleHandle*)a->MxObj;
    MxParticleHandle *bh = (MxParticleHandle*)b->MxObj;
    return MxBind::particles((MxPotential*)p->MxObj, ah->part(), bh->part());
}

HRESULT MxCBind_types(struct MxPotentialHandle *p, struct MxParticleTypeHandle *a, struct MxParticleTypeHandle *b, bool bound) {
    MXCBIND_CHECKHANDLE(p);
    MXCBIND_CHECKHANDLE(a);
    MXCBIND_CHECKHANDLE(b);
    return MxBind::types((MxPotential*)p->MxObj, (MxParticleType*)a->MxObj, (MxParticleType*)b->MxObj, bound);
}

HRESULT MxCBind_boundaryConditions(struct MxPotentialHandle *p, struct MxParticleTypeHandle *t) {
    MXCBIND_CHECKHANDLE(p);
    MXCBIND_CHECKHANDLE(t);
    return MxBind::boundaryConditions((MxPotential*)p->MxObj, (MxParticleType*)t->MxObj);
}

HRESULT MxCBind_boundaryCondition(struct MxPotentialHandle *p, struct MxBoundaryConditionHandle *bc, struct MxParticleTypeHandle *t) {
    MXCBIND_CHECKHANDLE(p);
    MXCBIND_CHECKHANDLE(bc);
    MXCBIND_CHECKHANDLE(t);
    return MxBind::boundaryCondition((MxPotential*)p->MxObj, (MxBoundaryCondition*)bc->MxObj, (MxParticleType*)t->MxObj);
}

HRESULT MxCBind_force(struct MxForceHandle *force, struct MxParticleTypeHandle *a_type) {
    MXCBIND_CHECKHANDLE(force);
    MXCBIND_CHECKHANDLE(a_type);
    return MxBind::force((MxForce*)force->MxObj, (MxParticleType*)a_type->MxObj);
}

HRESULT MxCBind_forceS(struct MxForceHandle *force, struct MxParticleTypeHandle *a_type, const char *coupling_symbol) {
    MXCBIND_CHECKHANDLE(force);
    MXCBIND_CHECKHANDLE(a_type);
    MXCPTRCHECK(coupling_symbol);
    return MxBind::force((MxForce*)force->MxObj, (MxParticleType*)a_type->MxObj, coupling_symbol);
}

HRESULT MxCBind_bonds(struct MxPotentialHandle *potential,
                      struct MxParticleListHandle *particles, 
                      double cutoff, 
                      struct MxParticleTypeHandle **ppairsA, 
                      struct MxParticleTypeHandle **ppairsB, 
                      unsigned int numTypes, 
                      double *half_life, 
                      double *bond_energy, 
                      struct MxBondHandleHandle **out, 
                      unsigned int *numOut) 
{
    MXCBIND_CHECKHANDLE(potential);
    MXCBIND_CHECKHANDLE(particles);

    MxPotential *_potential = (MxPotential*)potential->MxObj;
    MxParticleList *_particles = (MxParticleList*)particles->MxObj;
    std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs = NULL;
    if(ppairsA && ppairsB) {
        std::vector<std::pair<MxParticleType*, MxParticleType*>* > _pairs;
        struct MxParticleTypeHandle *pta, *ptb;
        for(unsigned int i = 0; i < numTypes; i++) {
            pta = ppairsA[i];
            ptb = ppairsB[i];
            MXCBIND_CHECKHANDLE(pta);
            MXCBIND_CHECKHANDLE(ptb);
            _pairs.push_back(new std::pair<MxParticleType*, MxParticleType*>(std::make_pair((MxParticleType*)pta->MxObj, (MxParticleType*)ptb->MxObj)));
        }
        pairs = &_pairs;
    }

    double _half_life = half_life ? *half_life : std::numeric_limits<double>::max();
    double _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<double>::max();

    HRESULT result;
    if(out && numOut) {
        std::vector<MxBondHandle*> *_outv;
        result = MxBind::bonds(_potential, _particles, cutoff, pairs, _half_life, _bond_energy, 0, &_outv);
        if(result == S_OK) {
            *numOut = _outv->size();
            MxBondHandleHandle *_out = (MxBondHandleHandle*)malloc(*numOut * sizeof(MxBondHandleHandle));
            for(unsigned int i = 0; i < _outv->size(); i++) {
                _out[i].MxObj = (void*)(*_outv)[i];
            }
            *out = _out;
            delete _outv;
        }
    }
    else {
        result = MxBind::bonds(_potential, _particles, cutoff, pairs, _half_life, _bond_energy);
    }

    if(pairs) {
        for(unsigned int i = 0; i < pairs->size(); i++) 
            delete (*pairs)[i];
    }

    return result;
}
