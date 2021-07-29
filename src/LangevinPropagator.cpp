/*
 * MeshDampedLangevinPropagator.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: andy
 */

#include <LangevinPropagator.h>
#include <MxModel.h>
#include "stochastic_rk.h"
#include <iostream>
#include <cstdlib>
#include <cstring>


LangevinPropagator::LangevinPropagator() {
}

// todo: continue here! Keep developing this structure and eliminate CObject-based nonsense
void LangevinPropagator::_ConstraintItems::update(const LangevinPropagator *prop) {
    if(!prop->mesh) return;

    this->args.clear();

    switch (this->actor->dataType())
    {
    case CONSTRAINABLE_TYPE::CONSTRAINABLE_CELL:
        for(CellPtr cell : prop->mesh->cells) {
            if(!cell->isRoot() && cell->constraintType == this->type) this->args.push_back(cell);
        }
        break;
    case CONSTRAINABLE_TYPE::CONSTRAINABLE_POLYGON:
        for(PolygonPtr poly : prop->mesh->polygons) {
            if(poly->constraintType == this->type) this->args.push_back(poly);
        }
        break;
    default:
        break;
    }

}

void LangevinPropagator::_ForceItems::update(const LangevinPropagator *prop) {
    if(!prop->mesh) return;

    this->args.clear();

    switch (this->actor->dataType())
    {
    case FORCABLE_TYPE::FORCABLE_CELL:
        for(CellPtr cell : prop->mesh->cells) {
            if(!cell->isRoot() && cell->forceType == this->type) this->args.push_back(cell);
        }
        break;
    case FORCABLE_TYPE::FORCABLE_POLYGON:
        for(PolygonPtr poly : prop->mesh->polygons) {
            if(poly->forceType == this->type) this->args.push_back(poly);
        }
    default:
        break;
    }

}

template<typename A, typename T, typename O>
HRESULT LangevinPropagator::updateItems(std::vector<LangevinPropagator::BaseItems<A, T, O> > &items)
{
    for(auto &i : items) {
        i.update(this);
    }
    return S_OK;
}

template<typename A, typename T, typename O> 
LangevinPropagator::BaseItems<A, T, O> &LangevinPropagator::getItem(std::vector<LangevinPropagator::BaseItems<A, T, O> >& items, A* key)
{
    auto it = std::find_if(
            items.begin(), items.end(),
            [key](const BaseItems<A, T, O>& x) { return x.actor == key;});
    if(it != items.end()) {
        return *it;
    }
    else {
        LangevinPropagator::BaseItems<A, T, O> item(key);
        items.push_back(item);
        return items.back();
    }
}

template<typename A, typename T, typename O>
HRESULT LangevinPropagator::bindTypeItem(std::vector<LangevinPropagator::BaseItems<A, T, O> >& items,
        A* key, T* type)
{
    LangevinPropagator::BaseItems<A, T, O> &ci = getItem(items, key);
    ci.type = type;
    ci.update(this);
    return S_OK;
}

HRESULT LangevinPropagator::setModel(MxModel *m) {
    this->model = m;
    this->mesh = m->mesh;
    m->propagator = this;

    return structureChanged();
}


HRESULT LangevinPropagator::step(MxReal dt) {

    HRESULT result = S_OK;

    resize();


    for(int i = 0; i < 10; ++i) {
        if((result = rungeKuttaStep(dt/10)) != S_OK) {
            return result;
        }
    }


    if((timeSteps % 20) == 0) {
        result = mesh->applyMeshOperations();
    }

    applyConstraints();

    timeSteps += 1;

    return result;
}

HRESULT LangevinPropagator::eulerStep(MxReal dt) {

    getPositions(dt, size, positions);

    getAccelerations(dt, size, positions, accel);

    for(int i = 0; i < size; ++i) {
        positions[i] = positions[i] + dt * accel[i];
    }

    setPositions(dt, size, positions);

    return S_OK;
}

HRESULT LangevinPropagator::rungeKuttaStep(MxReal dt)
{
    getAccelerations(dt, size, nullptr, k1);

    getPositions(dt, size, posInit);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k1[i] / 2.0 ;
    }

    getAccelerations(dt, size, positions, k2);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k2[i] / 2.0 ;
    }

    getAccelerations(dt, size, positions, k3);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt * k3[i];
    }

    getAccelerations(dt, size, positions, k4);

    for(int i = 0; i < size; ++i) {
        positions[i] = posInit[i] + dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
    }

    setPositions(dt, size, positions);

    return S_OK;
}

void LangevinPropagator::resize()
{
    if(!mesh) {
        return;
    }
    
    if(size != mesh->vertices.size()) {
        size = mesh->vertices.size();
        positions = (Vector3*)std::realloc(positions, size * sizeof(Vector3));
        accel = (Vector3*)std::realloc(accel, size * sizeof(Vector3));
        masses = (float*)std::realloc(masses, size * sizeof(float));

        posInit = (Vector3*)std::realloc(posInit, size * sizeof(Vector3));
        k1 = (Vector3*)std::realloc(k1, size * sizeof(Vector3));
        k2 = (Vector3*)std::realloc(k2, size * sizeof(Vector3));
        k3 = (Vector3*)std::realloc(k3, size * sizeof(Vector3));
        k4 = (Vector3*)std::realloc(k4, size * sizeof(Vector3));
    }

    uint32_t ssCount = 0;
    model->getStateVector(nullptr, &ssCount);
    if(stateVectorSize != ssCount) {
        stateVectorSize = ssCount;
        stateVector = (float*)std::realloc(stateVector, stateVectorSize * sizeof(float));
        stateVectorInit = (float*)std::realloc(stateVectorInit, stateVectorSize * sizeof(float));
        stateVectorK1 = (float*)std::realloc(stateVectorK1, stateVectorSize * sizeof(float));
        stateVectorK2 = (float*)std::realloc(stateVectorK2, stateVectorSize * sizeof(float));
        stateVectorK3 = (float*)std::realloc(stateVectorK3, stateVectorSize * sizeof(float));
        stateVectorK4 = (float*)std::realloc(stateVectorK4, stateVectorSize * sizeof(float));
    }
}

HRESULT LangevinPropagator::getAccelerations(float time, uint32_t len,
        const Vector3* pos, Vector3* acc)
{
    HRESULT result;

    if(len != mesh->vertices.size()) {
        return E_FAIL;
    }

    if(pos) {
        if(!SUCCEEDED(result = mesh->setPositions(len, pos))) {
            return result;
        }
    }

    VERIFY(applyForces());

    for(int i = 0; i < mesh->vertices.size(); ++i) {
        VertexPtr v = mesh->vertices[i];

        acc[i] = v->force;
    }

    return S_OK;
}

HRESULT LangevinPropagator::getPositions(float time, uint32_t len, Vector3* pos)
{
    for(int i = 0; i < len; ++i) {
        pos[i] = mesh->vertices[i]->position;
    }
    return S_OK;
}

HRESULT LangevinPropagator::applyConstraints()
{
    float sumError = 0;
    int iter = 0;

    do {

        for(ConstraintItems &ci : constraints) {
            ci.actor->project(ci.args);
        }
        
        iter += 1;

    } while(iter < 2);
    return S_OK;
}

HRESULT LangevinPropagator::stateVectorStep(MxReal dt)
{
    uint32_t count;
    model->getStateVector(stateVectorInit, &count);
    model->getStateVectorRate(dt, stateVector, stateVectorK1);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK1[i] / 2.0 ;
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK2);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK2[i] / 2.0 ;
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK3);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt * stateVectorK3[i];
    }

    model->getStateVectorRate(dt, stateVector, stateVectorK4);

    for(int i = 0; i < count; ++i) {
        stateVector[i] = stateVectorInit[i] + dt / 6. * (
                stateVectorK1[i] +
                2. * stateVectorK2[i] +
                2. * stateVectorK3[i] +
                stateVectorK4[i]
            );
    }

    model->setStateVector(stateVector);

    return S_OK;
}

HRESULT LangevinPropagator::structureChanged()
{
    if(!model) {
        return S_OK;
    }
    
    mesh = model->mesh;
    
    resize();

    VERIFY(updateItems(forces));

    VERIFY(updateItems(constraints));

    return S_OK;
}


HRESULT LangevinPropagator::bindForce(IForce* force, MxForcableType* type)
{
    if(type) {
        for(auto f : type->forces) {
            if(f == force) return S_OK;
        }
        type->forces.push_back(force);
        return bindTypeItem(forces, force, type);
    }
    return S_OK;
}

HRESULT LangevinPropagator::bindConstraint(IConstraint* constraint, MxConstrainableType* type)
{
    if(type) {
        for(auto c : type->constraints) {
            if(c == constraint) return S_OK;
        }
        type->constraints.push_back(constraint);
        return bindTypeItem(constraints, constraint, type);
    }
    return S_OK;
}

HRESULT MxBind_PropagatorModel(LangevinPropagator* propagator, MxModel* model)
{
    model->propagator = propagator;
    return propagator->setModel(model);
}

HRESULT LangevinPropagator::unbindConstraint(IConstraint* constraint)
{
	for(auto c : constraints) {
        if(c.actor == constraint) {
            for(auto tc = c.type->constraints.begin(); tc != c.type->constraints.end(); ++tc) {
                if(*tc == constraint) c.type->constraints.erase(tc);
            }
            c.unbind();
        }
    }
    return S_OK;
}

HRESULT LangevinPropagator::unbindForce(IForce* force)
{
	for(auto f : forces) {
        if(f.actor == force) {
            for(auto tf = f.type->forces.begin(); tf != f.type->forces.end(); ++tf) {
                if(*tf == force) f.type->forces.erase(tf);
            }
            f.unbind();
        }
    }
    return S_OK;
}

HRESULT LangevinPropagator::setPositions(float time, uint32_t len, const Vector3* pos)
{
    return mesh->setPositions(len, pos);
}

HRESULT LangevinPropagator::applyForces()
{
    for(ForceItems &f : forces) {
        f.actor->applyForce(0, f.args);
    }

    return S_OK;
}
