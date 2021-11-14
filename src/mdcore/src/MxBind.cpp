/*
 * MxBind.cpp
 *
 *  Created on: Feb 13, 2021
 *      Author: andy
 */

#include <MxBind.hpp>
#include <MxParticle.h>
#include <engine.h>
#include <string>
#include "../../state/MxSpeciesList.h"
#include "../../mx_error.h"
#include "../../MxLogger.h"
#include "../../MxUtil.h"


HRESULT universe_bind_potential(MxPotential *p, MxParticle *a, MxParticle *b) {
    if (!a || !b) return S_OK;

    Log(LOG_DEBUG) << p->name << ", " << a->id << ", " << b->id;

    auto bond = new MxBondHandle();
    bond->init(p, a->py_particle(), b->py_particle());
    return S_OK;
}

HRESULT universe_bind_potential(MxPotential *p, MxParticleType *a, MxParticleType *b, bool bound) {
    if (!a || !b) return S_OK;

    Log(LOG_DEBUG) << p->name << ", " << a->name << ", " << b->name << ", " << bound;

    MxPotential *pot = NULL;

    if(p->create_func) {
        pot = p->create_func(p, a, b);
    }
    else {
        pot = p;
    }

    if(bound) {
        pot->flags = pot->flags | POTENTIAL_BOUND;
    }

    if(engine_addpot(&_Engine, pot, a->id, b->id) != engine_err_ok) {
        std::string msg = "failed to add potential to engine: error ";
        msg += std::to_string(-engine_err);
        msg += ", ";
        msg += engine_err_msg[-engine_err];
        Log(LOG_CRITICAL) << msg;
        auto msg_c = msg.c_str();
        return mx_error(E_FAIL, msg_c);
    }
    
    return S_OK;
}

HRESULT universe_bind_potential_cuboid(MxPotential *p, MxParticleType *t) {
    Log(LOG_DEBUG) << p->name << ", " << t->name;
    return engine_add_cuboid_potential(&_Engine, p, t->id);
}

HRESULT universe_bind_potential(MxPotential *p, MxBoundaryConditions *bcs, MxParticleType *t) {
    Log(LOG_DEBUG) << p->name << ", " << t->name;
    bcs->set_potential(t, p);
    return S_OK;
}

HRESULT universe_bind_potential(MxPotential *p, MxBoundaryCondition *bc, MxParticleType *t) {
    Log(LOG_DEBUG) << p->name << ", " << t->name;
    bc->set_potential(t, p);
    return S_OK;
}

HRESULT universe_bind_force(MxForce *force, MxParticleType *a_type, const std::string* coupling_symbol) {
    std::string msg = a_type->name;
    std::string msg_coupling_symbol = coupling_symbol ? ", " + *coupling_symbol : "";
    Log(LOG_DEBUG) << msg + msg_coupling_symbol;
    
    if(coupling_symbol == NULL) {
        if(engine_add_singlebody_force(&_Engine, force, a_type->id, -1) != engine_err_ok) {
            std::string msg = "failed to add force to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }
    
    if(a_type->species) {
        int index = a_type->species->index_of(coupling_symbol->c_str());
        if(index < 0) {
            std::string msg = "could not bind force, the particle type ";
            msg += a_type->name;
            msg += " has a chemical species state vector, but it does not have the symbol ";
            msg += *coupling_symbol;
            Log(LOG_CRITICAL) << msg;
            return mx_error(E_FAIL, msg.c_str());
        }
        
        if(engine_add_singlebody_force(&_Engine, force, a_type->id, index) != engine_err_ok) {
            std::string msg = "failed to add force to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            Log(LOG_CRITICAL) << msg;
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }
    else {
        std::string msg = "could not add force, given a coupling symbol, but the particle type ";
        msg += a_type->name;
        msg += " does not have a chemical species vector";
        Log(LOG_CRITICAL) << msg;
        return mx_error(E_FAIL, msg.c_str());
    }

}

HRESULT MxBind::particles(MxPotential *p, MxParticle *a, MxParticle *b) {
    return universe_bind_potential(p, a, b);
}

HRESULT MxBind::types(MxPotential *p, MxParticleType *a, MxParticleType *b, bool bound) {
    return universe_bind_potential(p, a, b, bound);
}

HRESULT MxBind::cuboid(MxPotential *p, MxParticleType *t) {
    return universe_bind_potential_cuboid(p, t);
}

HRESULT MxBind::boundaryConditions(MxPotential *p, MxBoundaryConditions *bcs, MxParticleType *t) {
    return universe_bind_potential(p, bcs, t);
}

HRESULT MxBind::boundaryCondition(MxPotential *p, MxBoundaryCondition *bc, MxParticleType *t) {
    return universe_bind_potential(p, bc, t);
}

HRESULT MxBind::force(MxForce *force, MxParticleType *a_type, const std::string* coupling_symbol) {
    return universe_bind_force(force, a_type, coupling_symbol);
}

HRESULT MxBind::bonds(MxPotential* potential,
                      MxParticleList *particles, 
                      const double &cutoff, 
                      std::vector<std::pair<MxParticleType*, MxParticleType*>* > *pairs, 
                      const double &half_life, 
                      const double &bond_energy, 
                      uint32_t flags, 
                      std::vector<MxBondHandle*> **out) 
{ 
    Log(LOG_DEBUG);
    auto result = MxBondHandle::pairwise(potential, particles, cutoff, pairs, half_life, bond_energy, flags);
    if (out) *out = result;
    return S_OK;
}

HRESULT MxBind::sphere(MxPotential *potential,
                        const int &n,
                        MxVector3f *center,
                        const float &radius,
                        std::pair<float, float> *phi, 
                        MxParticleType *type, 
                        std::pair<MxParticleList*, std::vector<MxBondHandle*>*> **out)
{
    Log(LOG_TRACE);

    static const float Pi = M_PI;


    //potential
    //*     number of subdivisions
    //*     tuple of starting / stopping theta (polar angle)
    //*     center of sphere
    //*     radius of sphere

    float phi0 = 0;
    float phi1 = Pi;

    if(phi) {
        phi0 = std::get<0>(*phi);
        phi1 = std::get<1>(*phi);

        if(phi0 < 0 || phi0 > Pi) mx_exp(std::logic_error("phi_0 must be between 0 and pi"));
        if(phi1 < 0 || phi1 > Pi) mx_exp(std::logic_error("phi_1 must be between 0 and pi"));
        if(phi1 < phi0) mx_exp(std::logic_error("phi_1 must be greater than phi_0"));
    }

    MxVector3f _center =  center ? *center : engine_center();

    std::vector<MxVector3f> vertices;
    std::vector<int32_t> indices;

    MxMatrix4f s = MxMatrix4f::scaling(MxVector3f{radius, radius, radius});
    MxMatrix4f t = MxMatrix4f::translation(_center);
    MxMatrix4f m = t * s;

    Mx_Icosphere(n, phi0, phi1, vertices, indices);

    MxVector3f velocity;

    MxParticleList *parts = new MxParticleList(vertices.size());
    parts->nr_parts = vertices.size();

    // Euler formula for graphs:
    // For a closed polygon -- non-manifold mesh: T−E+V=1 -> E = T + V - 1
    // for a sphere: T−E+V=2. -> E = T + V - 2

    int edges;
    if(phi0 <= 0 && phi1 >= Pi) {
        edges = vertices.size() + (indices.size() / 3) - 2;
    }
    else if(mx::almost_equal(phi0, 0.0f) || mx::almost_equal(phi1, Pi)) {
        edges = vertices.size() + (indices.size() / 3) - 1;
    }
    else {
        edges = vertices.size() + (indices.size() / 3);
    }

    if(edges <= 0) return E_FAIL;

    std::vector<MxBondHandle*> *bonds = new std::vector<MxBondHandle*>();

    for(int i = 0; i < vertices.size(); ++i) {
        MxVector3f pos = m.transformPoint(vertices[i]);
        MxParticleHandle *p = (*type)(&pos, &velocity);
        parts->parts[i] = p->id;
    }

    if(vertices.size() > 0 && indices.size() == 0) return E_FAIL;

    int nbonds = 0;
    for(int i = 0; i < indices.size(); i += 3) {
        int a = indices[i];
        int b = indices[i+1];
        int c = indices[i+2];

        nbonds += insert_bond(*bonds, a, b, potential, parts);
        nbonds += insert_bond(*bonds, b, c, potential, parts);
        nbonds += insert_bond(*bonds, c, a, potential, parts);
    }

    if(nbonds != bonds->size()) {
        std::string msg = "unknown error in finding edges for sphere mesh, \n";
        msg += "vertices: " + std::to_string(vertices.size()) + "\n";
        msg += "indices: " + std::to_string(indices.size()) + "\n";
        msg += "expected edges: " + std::to_string(edges) + "\n";
        msg += "found edges: " + std::to_string(nbonds);
        mx_exp(std::overflow_error(msg));
    }

    if(out) *out = new std::pair<MxParticleList*, std::vector<MxBondHandle*>*>(parts, bonds);

    return S_OK;
}
