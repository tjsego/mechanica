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
#include <../../io/MxFIO.h>
#include <iostream>
#include <random>
#include <../../state/MxStateVector.h>
#include "../../state/MxSpeciesList.h"
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
{
    type = FORCE_CONSTANTPY;
}

MxConstantForcePy::MxConstantForcePy(const MxVector3f &f, const float &period) : 
    MxConstantForce(f, period)
{
    type = FORCE_CONSTANTPY;
    callable = NULL;
}

MxConstantForcePy::MxConstantForcePy(PyObject *f, const float &period) : 
    MxConstantForce(), 
    callable(f)
{
    type = FORCE_CONSTANTPY;

    setPeriod(period);
    if(PyList_Check(f)) {
        MxVector3f fv = mx::cast<PyObject, MxVector3f>(f);
        callable = NULL;
        MxConstantForce::setValue(fv);
    }
    else if(callable) {
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

HRESULT MxForce::bind_species(MxParticleType *a_type, const std::string &coupling_symbol) {
    std::string msg = a_type->name;
    Log(LOG_DEBUG) << msg + coupling_symbol;

    if(a_type->species) {
        int index = a_type->species->index_of(coupling_symbol.c_str());
        if(index < 0) {
            std::string msg = "could not bind force, the particle type ";
            msg += a_type->name;
            msg += " has a chemical species state vector, but it does not have the symbol ";
            msg += coupling_symbol;
            Log(LOG_CRITICAL) << msg;
            return mx_error(E_FAIL, msg.c_str());
        }

        this->stateVectorIndex = index;
    }
    else {
        std::string msg = "could not add force, given a coupling symbol, but the particle type ";
        msg += a_type->name;
        msg += " does not have a chemical species vector";
        Log(LOG_CRITICAL) << msg;
        return mx_error(E_FAIL, msg.c_str());
    }

    return S_OK;
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

void eval_sum_force(struct MxForce *force, struct MxParticle *p, FPTYPE *f) {
    MxForceSum *sf = (MxForceSum*)force;
    
    std::vector<FPTYPE> f1(3, 0.0), f2(3, 0.0);
    (*sf->f1->func)(sf->f1, p, f1.data());
    (*sf->f2->func)(sf->f2, p, f2.data());

    float scaling = scaling_constant(p, force->stateVectorIndex);
    for(unsigned int i = 0; i < 3; i++) 
        p->f[i] += scaling * (f1[i] + f2[i]);
}

MxForce *MxForce_add(MxForce *f1, MxForce *f2) {
    MxForceSum *sf = new MxForceSum();
    sf->func = (MxForce_EvalFcn)eval_sum_force;

    sf->f1 = f1;
    sf->f2 = f2;
    
    return (MxForce*)sf;
}

MxForce& MxForce::operator+(const MxForce& rhs) {
    return *MxForce_add(this, const_cast<MxForce*>(&rhs));
}

std::string MxForce::toString() {
    MxIOElement *fe = new MxIOElement();
    MxMetaData metaData;

    if(mx::io::toFile(this, metaData, fe) != S_OK) 
        return "";

    return mx::io::toStr(fe, metaData);
}

MxForce *MxForce::fromString(const std::string &str) {
    return mx::io::fromString<MxForce*>(str);
}

/**
 * Implements a force:
 *
 * f_b = p / tau * ((T_0 / T) - 1)
 */
static void berendsen_force(Berendsen *t, MxParticle *p, FPTYPE *f) {
    MxParticleType *type = &engine::types[p->typeId];

    if(type->kinetic_energy <= 0 || type->target_energy <= 0) return;

    float scale = t->itau * ((type->target_energy / type->kinetic_energy) - 1.0) * scaling_constant(p, t->stateVectorIndex);
    f[0] += scale * p->v[0];
    f[1] += scale * p->v[1];
    f[2] += scale * p->v[2];
}

static void constant_force(MxConstantForce *cf, MxParticle *p, FPTYPE *f) {
    float scale = scaling_constant(p, cf->stateVectorIndex);
    f[0] += cf->force[0] * scale;
    f[1] += cf->force[1] * scale;
    f[2] += cf->force[2] * scale;
}


/**
 * Implements a friction force:
 *
 * f_f = - || v || / tau * v
 */
static void friction_force(Friction *t, MxParticle *p, FPTYPE *f) {
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }
    
    float v2 = p->velocity.dot();
    float scale = -1. * t->coef * v2  * scaling_constant(p, t->stateVectorIndex);
    
    f[0] += scale * p->v[0] + p->persistent_force[0];
    f[1] += scale * p->v[1] + p->persistent_force[1];
    f[2] += scale * p->v[2] + p->persistent_force[2];
}

static void gaussian_force(Gaussian *t, MxParticle *p, FPTYPE *f) {
    
    if((_Engine.integrator_flags & INTEGRATOR_UPDATE_PERSISTENTFORCE) &&
       (_Engine.time + p->id) % t->durration_steps == 0) {
        
        p->persistent_force = MxRandomVector(t->mean, t->std);
    }

    float scale = scaling_constant(p, t->stateVectorIndex);
    
    f[0] += scale * p->persistent_force[0];
    f[1] += scale * p->persistent_force[1];
    f[2] += scale * p->persistent_force[2];
}

Berendsen *berenderson_create(float tau) {
    auto *obj = new Berendsen();

    obj->type = FORCE_BERENDSEN;
    obj->func = (MxForce_EvalFcn)berendsen_force;
    obj->itau = 1/tau;

    return obj;
}

Gaussian *random_create(float mean, float std, float durration) {
    auto *obj = new Gaussian();
    
    obj->type = FORCE_GAUSSIAN;
    obj->func = (MxForce_EvalFcn)gaussian_force;
    obj->std = std;
    obj->mean = mean;
    obj->durration_steps = std::ceil(durration / _Engine.dt);
    
    return obj;
}

Friction *friction_create(float coef, float mean, float std, float durration) {
    auto *obj = new Friction();
    
    obj->type = FORCE_FRICTION;
    obj->func = (MxForce_EvalFcn)friction_force;
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
    type = FORCE_CONSTANT;
    func = (MxForce_EvalFcn)constant_force;
}

MxConstantForce::MxConstantForce(const MxVector3f &f, const float &period) : MxConstantForce() {
    type = FORCE_CONSTANT;
    updateInterval = period;
    setValue(f);
}

MxConstantForce::MxConstantForce(MxUserForceFuncType *f, const float &period) : MxConstantForce() {
    type = FORCE_CONSTANT;
    updateInterval = period;
    setValue(f);
}


MxForceSum *MxForceSum::fromForce(MxForce *f) {
    if(f->type != FORCE_SUM) 
        return 0;
    return (MxForceSum*)f;
}

MxConstantForce *MxConstantForce::fromForce(MxForce *f) {
    if(f->type != FORCE_CONSTANT) 
        return 0;
    return (MxConstantForce*)f;
}

MxConstantForcePy *MxConstantForcePy::fromForce(MxForce *f) {
    if(f->type != FORCE_CONSTANTPY) 
        return 0;
    return (MxConstantForcePy*)f;
}

Berendsen *Berendsen::fromForce(MxForce *f) {
    if(f->type != FORCE_BERENDSEN) 
        return 0;
    return (Berendsen*)f;
}

Gaussian *Gaussian::fromForce(MxForce *f) {
    if(f->type != FORCE_GAUSSIAN) 
        return 0;
    return (Gaussian*)f;
}

Friction *Friction::fromForce(MxForce *f) {
    if(f->type != FORCE_FRICTION) 
        return 0;
    return (Friction*)f;
}


namespace mx { namespace io {

#define MXFORCEIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXFORCEIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MXFORCE_TYPE &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    fileElement->type = "FORCE_TYPE";
    fileElement->value = std::to_string((unsigned int)dataElement);

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MXFORCE_TYPE *dataElement) {

    unsigned int ui;
    if(fromFile(fileElement, metaData, &ui) != S_OK) 
        return E_FAIL;

    *dataElement = (MXFORCE_TYPE)ui;

    return S_OK;
}

template <>
HRESULT toFile(const MxConstantForce &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEIOTOEASY(fe, "updateInterval", dataElement.updateInterval);
    MXFORCEIOTOEASY(fe, "lastUpdate", dataElement.lastUpdate);
    MXFORCEIOTOEASY(fe, "force", dataElement.force);

    fileElement->type = "ConstantForce";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxConstantForce *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "updateInterval", &dataElement->updateInterval);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "lastUpdate", &dataElement->lastUpdate);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "force", &dataElement->force);
    dataElement->userFunc = NULL;

    return S_OK;
}

template <>
HRESULT toFile(const MxConstantForcePy &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEIOTOEASY(fe, "updateInterval", dataElement.updateInterval);
    MXFORCEIOTOEASY(fe, "lastUpdate", dataElement.lastUpdate);
    MXFORCEIOTOEASY(fe, "force", dataElement.force);

    fileElement->type = "ConstantPyForce";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxConstantForcePy *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "updateInterval", &dataElement->updateInterval);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "lastUpdate", &dataElement->lastUpdate);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "force", &dataElement->force);
    dataElement->userFunc = NULL;
    dataElement->callable = NULL;

    return S_OK;
}

template <>
HRESULT toFile(const MxForceSum &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);

    if(dataElement.f1 != NULL) 
        MXFORCEIOTOEASY(fe, "Force1", dataElement.f1);
    if(dataElement.f2 != NULL) 
        MXFORCEIOTOEASY(fe, "Force2", dataElement.f2);

    fileElement->type = "SumForce";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxForceSum *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    
    feItr = fileElement.children.find("Force1");
    if(feItr != fileElement.children.end()) {
        dataElement->f1 = NULL;
        if(fromFile(*feItr->second, metaData, &dataElement->f1) != S_OK) 
            return E_FAIL;
    }

    feItr = fileElement.children.find("Force2");
    if(feItr != fileElement.children.end()) {
        dataElement->f2 = NULL;
        if(fromFile(*feItr->second, metaData, &dataElement->f2) != S_OK) 
            return E_FAIL;
    }

    return S_OK;
}

template <>
HRESULT toFile(const Berendsen &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEIOTOEASY(fe, "itau", dataElement.itau);

    fileElement->type = "BerendsenForce";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Berendsen *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "itau", &dataElement->itau);
    dataElement->func = (MxForce_EvalFcn)berendsen_force;

    return S_OK;
}

template <>
HRESULT toFile(const Gaussian &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEIOTOEASY(fe, "std", dataElement.std);
    MXFORCEIOTOEASY(fe, "mean", dataElement.mean);
    MXFORCEIOTOEASY(fe, "durration_steps", dataElement.durration_steps);
    
    fileElement->type = "GaussianForce";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Gaussian *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "std", &dataElement->std);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "mean", &dataElement->mean);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "durration_steps", &dataElement->durration_steps);
    dataElement->func = (MxForce_EvalFcn)gaussian_force;

    return S_OK;
}

template <>
HRESULT toFile(const Friction &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement.type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement.stateVectorIndex);
    MXFORCEIOTOEASY(fe, "coef", dataElement.coef);
    MXFORCEIOTOEASY(fe, "std", dataElement.std);
    MXFORCEIOTOEASY(fe, "mean", dataElement.mean);
    MXFORCEIOTOEASY(fe, "durration_steps", dataElement.durration_steps);

    fileElement->type = "FrictionForce";
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Friction *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &dataElement->type);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "coef", &dataElement->coef);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "std", &dataElement->std);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "mean", &dataElement->mean);
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "durration_steps", &dataElement->durration_steps);
    dataElement->func = (MxForce_EvalFcn)friction_force;

    return S_OK;
}

template <>
HRESULT toFile(MxForce *dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 

    if(dataElement->type & FORCE_BERENDSEN) 
        return toFile(*(Berendsen*)dataElement, metaData, fileElement);
    else if(dataElement->type & FORCE_CONSTANT) 
        return toFile(*(MxConstantForce*)dataElement, metaData, fileElement);
    else if(dataElement->type & FORCE_CONSTANTPY) 
        return toFile(*(MxConstantForcePy*)dataElement, metaData, fileElement);
    else if(dataElement->type & FORCE_FRICTION) 
        return toFile(*(Friction*)dataElement, metaData, fileElement);
    else if(dataElement->type & FORCE_GAUSSIAN) 
        return toFile(*(Gaussian*)dataElement, metaData, fileElement);
    else if(dataElement->type & FORCE_SUM) 
        return toFile(*(MxForceSum*)dataElement, metaData, fileElement);
    
    MxIOElement *fe;

    MXFORCEIOTOEASY(fe, "type", dataElement->type);
    MXFORCEIOTOEASY(fe, "stateVectorIndex", dataElement->stateVectorIndex);
    
    fileElement->type = "Force";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxForce **dataElement) {

    MxIOChildMap::const_iterator feItr;

    MXFORCE_TYPE fType;
    MXFORCEIOFROMEASY(feItr, fileElement.children, metaData, "type", &fType);

    if(fType & FORCE_BERENDSEN) {
        Berendsen *f = new Berendsen();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    else if(fType & FORCE_CONSTANT) {
        MxConstantForce *f = new MxConstantForce();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    else if(fType & FORCE_CONSTANTPY) {
        MxConstantForcePy *f = new MxConstantForcePy();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    else if(fType & FORCE_FRICTION) {
        Friction *f = new Friction();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    else if(fType & FORCE_GAUSSIAN) {
        Gaussian *f = new Gaussian();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    else if(fType & FORCE_SUM) {
        MxForceSum *f = new MxForceSum();
        if(fromFile(fileElement, metaData, f) != S_OK) 
            return E_FAIL;
        *dataElement = f;
        return S_OK;
    }
    
    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::vector<MxForce*> *dataElement) {
    unsigned int numEls = fileElement.children.size();
    dataElement->reserve(numEls);
    for(unsigned int i = 0; i < numEls; i++) {
        MxForce *de = NULL;
        auto itr = fileElement.children.find(std::to_string(i));
        if(itr == fileElement.children.end()) 
            return E_FAIL;
        if(fromFile(*itr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        dataElement->push_back(de);
    }
    return S_OK;
}

}};

MxForceSum *MxForceSum_fromStr(const std::string &str) {
    return (MxForceSum*)MxForce::fromString(str);
}

Berendsen *Berendsen_fromStr(const std::string &str)  {
    return (Berendsen*)MxForce::fromString(str);
}

Gaussian *Gaussian_fromStr(const std::string &str) {
    return (Gaussian*)MxForce::fromString(str);
}

Friction *Friction_fromStr(const std::string &str) {
    return (Friction*)MxForce::fromString(str);
}
