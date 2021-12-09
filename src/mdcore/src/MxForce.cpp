/*
 * MxForce.cpp
 *
 *  Created on: May 21, 2020
 *      Author: andy
 */

#include <MxForce.h>
#include <engine.h>
#include <MxParticle.h>
#include <../../mx_error.h>
#include <../../MxLogger.h>
#include <iostream>
#include <random>
#include <../../state/MxStateVector.h>
#include <MxPy.h>

static Berendsen *berenderson_create(float tau);
static Gaussian *random_create(float mean, float std, float durration);
static Friction *friction_create(float coef, float mean, float std, float durration);

static float scaling_constant(MxParticle *part, int stateVectorIndex) {
    if(part->state_vector && stateVectorIndex >= 0) {
        return part->state_vector->fvec[stateVectorIndex];
    }
    else {
        return 1.f;
    }
}

MxVector3f MxConstantForce::getValue() {
    if(userFunc) return (*userFunc)(this);
    return force;
}

void MxConstantForce::setValue(const MxVector3f &f) {
    force = f;
}

void MxConstantForce::setValue(MxUserForceFuncType *_userFunc) {
    if(_userFunc) userFunc = _userFunc;
    setValue((*userFunc)(this));
}

float MxConstantForce::getPeriod() {
    return updateInterval;
}

void MxConstantForce::setPeriod(const float &period) {
    updateInterval = period;
}

MxVector3f pyConstantForceFunction(PyObject *callable) {
    Log(LOG_TRACE);

    PyObject *result = PyObject_CallObject(callable, NULL);

    if(result == NULL) {
        PyObject *err = PyErr_Occurred();
        Log(LOG_CRITICAL) << pyerror_str();
        PyErr_Clear();
        return MxVector3f();
    }
    MxVector3f out = mx::cast<PyObject, MxVector3f>(result);
    Py_DECREF(result);
    return out;
}

MxConstantForcePy::MxConstantForcePy() : 
    MxConstantForce() 
{}

MxConstantForcePy::MxConstantForcePy(const MxVector3f &f, const float &period) : 
    MxConstantForce(f, period)
{}

MxConstantForcePy::MxConstantForcePy(PyObject *f, const float &period) : 
    MxConstantForce(), 
    callable(f)
{
    setPeriod(period);
    if(callable) {
        Py_IncRef(callable);
    }
}

MxConstantForcePy::~MxConstantForcePy(){
    if(callable) Py_DecRef(callable);
}

void MxConstantForcePy::onTime(double time)
{
    if(callable && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue(callable);
    }
}

MxVector3f MxConstantForcePy::getValue() {
    if(callable && callable != Py_None) return pyConstantForceFunction(callable);
    return force;
}

void MxConstantForcePy::setValue(PyObject *_userFunc) {
    if(_userFunc) callable = _userFunc;
    if(callable && callable != Py_None) MxConstantForce::setValue(getValue());
}

Berendsen* MxForce::berenderson_tstat(const float &tau) {
    Log(LOG_DEBUG);

    try {
        return berenderson_create(tau);
    }
    catch (const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

Gaussian* MxForce::random(const float &std, const float &mean, const float &duration) {
    Log(LOG_DEBUG);

    try {
        return random_create(mean, std, duration);
    }
    catch (const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

Friction* MxForce::friction(const float &coef, const float &std, const float &mean, const float &duration) {
    Log(LOG_DEBUG);

    try {
        return friction_create(coef, std, mean, duration);
    }
    catch (const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

void eval_sum_force(struct MxForce *force, struct MxParticle *p, int stateVectorId, FPTYPE *f) {
    MxForceSum *sf = (MxForceSum*)force;
    (*sf->f1->func)(sf->f1, p, stateVectorId, f);
    (*sf->f2->func)(sf->f2, p, stateVectorId, f);
}

MxForce *MxForce_add(MxForce *f1, MxForce *f2) {
    MxForceSum *sf = new MxForceSum();
    sf->func = (MxForce_OneBodyPtr)eval_sum_force;

    sf->f1 = f1;
    sf->f2 = f2;
    
    return (MxForce*)sf;
}

MxForce& MxForce::operator+(const MxForce& rhs) {
    return *MxForce_add(this, const_cast<MxForce*>(&rhs));
}

/**
 * Implements a force:
 *
 * f_b = p / tau * ((T_0 / T) - 1)
 */
static void berendsen_force(Berendsen* t, MxParticle *p, int stateVectorIndex, FPTYPE*f) {
    MxParticleType *type = &engine::types[p->typeId];

    if(type->kinetic_energy <= 0 || type->target_energy <= 0) return;

    float scale = t->itau * ((type->target_energy / type->kinetic_energy) - 1.0);
    f[0] += scale * p->v[0];
    f[1] += scale * p->v[1];
    f[2] += scale * p->v[2];
}

static void constant_force(MxConstantForce* cf, MxParticle *p, int stateVectorIndex, FPTYPE*f) {
    float scale = scaling_constant(p, stateVectorIndex);
    f[0] += cf->force[0] * scale;
    f[1] += cf->force[1] * scale;
    f[2] += cf->force[2] * scale;
}


/**
 * Implements a friction force:
 *
 * f_f = - || v || / tau * v
 */
static void friction_force(Friction* t, MxParticle *p, int stateVectorIndex, FPTYPE*f) {
    MxParticleType *type = &engine::types[p->typeId];
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }
    
    float v2 = p->velocity.dot();
    float scale = -1. * t->coef * v2;
    
    f[0] += scale * p->v[0] + p->persistent_force[0];
    f[1] += scale * p->v[1] + p->persistent_force[1];
    f[2] += scale * p->v[2] + p->persistent_force[2];
}

static void gaussian_force(Gaussian* t, MxParticle *p, int stateVectorIndex, FPTYPE*f) {
    MxParticleType *type = &engine::types[p->typeId];
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }
    
    f[0] += p->persistent_force[0];
    f[1] += p->persistent_force[1];
    f[2] += p->persistent_force[2];
}

Berendsen *berenderson_create(float tau) {
    auto *obj = new Berendsen();

    obj->func = (MxForce_OneBodyPtr)berendsen_force;
    obj->itau = 1/tau;

    return obj;
}

Gaussian *random_create(float mean, float std, float durration) {
    auto *obj = new Gaussian();
    
    obj->func = (MxForce_OneBodyPtr)gaussian_force;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return obj;
}

Friction *friction_create(float coef, float mean, float std, float durration) {
    auto *obj = new Friction();
    
    obj->func = (MxForce_OneBodyPtr)friction_force;
    obj->coef = coef;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return obj;
}

void MxConstantForce::onTime(double time)
{
    if(userFunc && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue((*userFunc)(this));
    }
}

MxConstantForce::MxConstantForce() {
    func = (MxForce_OneBodyPtr)constant_force;
}

MxConstantForce::MxConstantForce(const MxVector3f &f, const float &period) : MxConstantForce() {
    updateInterval = period;
    setValue(f);
}

MxConstantForce::MxConstantForce(MxUserForceFuncType *f, const float &period) : MxConstantForce() {
    updateInterval = period;
    setValue(f);
}
