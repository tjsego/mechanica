/*
 * MxUniverse.cpp
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */
#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <MxUniverse.h>
#include <MxForce.h>
#include <MxPy.h>
#include <MxSimulator.h>
#include <MxConvert.hpp>
#include <MxUtil.h>
#include <metrics.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <MxThreadPool.hpp>
#include <MxCuboid.hpp>
#include <MxBind.hpp>
#include <MxPy.h>
#include <state/MxStateVector.h>
#include <state/MxSpeciesList.h>
#include <MxSystem.h>
#include <mx_error.h>

MxUniverse Universe = {
    .isRunning = false
};

MxUniverse *getUniverse() {
    return &Universe;
}

// the single static engine instance per process

// complete and total hack to get the global engine to show up here
// instead of the mdcore static lib.
// TODO: fix this crap.
engine _Engine = {
        .flags = 0
};

// default to paused universe
static uint32_t universe_flags = 0;


CAPI_FUNC(struct engine*) engine_get()
{
    return &_Engine;
}


// TODO: fix error handling values
#define UNIVERSE_CHECKERROR() { \
    if (_Engine.flags == 0 ) { \
        std::string err = "Error in "; \
        err += MX_FUNCTION; \
        err += ", Universe not initialized"; \
        return mx_error(E_FAIL, err.c_str()); \
    } \
    }

#define UNIVERSE_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = MX_FUNCTION; \
            err += "universe not initialized"; \
            throw std::domain_error(err.c_str()); \
        }

#define UNIVERSE_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define UNIVERSE_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        mx_exp(e); return retval; \
    }

MxUniverseConfig::MxUniverseConfig() :
    origin {0, 0, 0},
    dim {10, 10, 10},
    spaceGridSize {4, 4, 4},
    cutoff{1},
    flags{0},
    maxTypes{64},
    dt{0.01},
    temp{1},
    nParticles{100},
    threads{mx::ThreadPool::hardwareThreadSize()},
    integrator{EngineIntegrator::FORWARD_EULER},
    boundaryConditionsPtr{new MxBoundaryConditionsArgsContainer()},
    max_distance{-1},
    timers_mask {0},
    timer_output_period {-1}
{
}

MxMatrix3f *MxUniverse::virial(MxVector3f *origin, float *radius, std::vector<MxParticleType*> *types) {
    try {
        MxVector3f _origin = origin ? *origin : MxUniverse::getCenter();
        float _radius = radius ? *radius : 2 * _origin.max();

        std::set<short int> typeIds;

        if (types) {
            for (auto type : *types) 
                if (type) 
                    typeIds.insert(type->id);
        }
        else {
            for(int i = 0; i < _Engine.nr_types; ++i) 
                typeIds.insert(i);
        }

        MxMatrix3f *m;
        if(SUCCEEDED(MxCalculateVirial(_origin.data(), _radius, typeIds, m->data()))) {
            return m;
        }
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
    return NULL;
}

HRESULT MxUniverse::step(const double &until, const double &dt) {
    UNIVERSE_TRY();
    return MxUniverse_Step(until, dt);
    UNIVERSE_FINALLY(1);
}

HRESULT MxUniverse::stop() {
    UNIVERSE_TRY();
    return MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, false);
    UNIVERSE_FINALLY(1);
}

HRESULT MxUniverse::start() {
    UNIVERSE_TRY();
    return MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, true);
    UNIVERSE_FINALLY(1);
}

HRESULT MxUniverse::reset() {
    UNIVERSE_TRY();
    return engine_reset(&_Engine);
    UNIVERSE_FINALLY(1);
}

MxParticleList *MxUniverse::particles() {
    UNIVERSE_TRY();
    return MxParticleList::all();
    UNIVERSE_FINALLY(NULL);
}

void MxUniverse::resetSpecies() {
    UNIVERSE_TRY();
    
    for(int i = 0; i < _Engine.s.nr_parts; ++i) {
        MxParticle *part = _Engine.s.partlist[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    for(int i = 0; i < _Engine.s.largeparts.count; ++i) {
        MxParticle *part = &_Engine.s.largeparts.parts[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    // redraw, state changed. 
    MxSimulator::get()->redraw();
    
    UNIVERSE_FINALLY();
}

std::vector<std::vector<std::vector<MxParticleList*> > > MxUniverse::grid(MxVector3i shape) {
    UNIVERSE_TRY();
    return MxParticle_Grid(shape);
    UNIVERSE_FINALLY(std::vector<std::vector<std::vector<MxParticleList*> > >());
}

std::vector<MxBondHandle*> *MxUniverse::bonds() {
    UNIVERSE_TRY();
    std::vector<MxBondHandle*> *bonds;

    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        MxBond *b = &_Engine.bonds[i];
        if (b->flags & BOND_ACTIVE)
            bonds->push_back(new MxBondHandle(i));
    }
    return bonds;
    UNIVERSE_FINALLY(NULL);
}


double MxUniverse::getTemperature() {
    return engine_temperature(&_Engine);
}

double MxUniverse::getTime() {
    return _Engine.time * _Engine.dt;
}

double MxUniverse::getDt() {
    return _Engine.dt;
}

MxEventList *MxUniverse::getEventList() {
    return (MxEventList *)_Engine.events;
}

MxBoundaryConditions *MxUniverse::getBoundaryConditions() {
    return &_Engine.boundary_conditions;
}

double MxUniverse::getKineticEnergy() {
    return engine_kinetic_energy(&_Engine);
}

MxVector3f MxUniverse::origin()
{
    return MxVector3f{(float)_Engine.s.origin[0], (float)_Engine.s.origin[1], (float)_Engine.s.origin[2]};
}

MxVector3f MxUniverse::dim()
{
    return MxVector3f{(float)_Engine.s.dim[0], (float)_Engine.s.dim[1], (float)_Engine.s.dim[2]};
}

CAPI_FUNC(HRESULT) MxUniverse_Step(double until, double dt) {

    if(engine_err != 0) {
        return E_FAIL;
    }

    if ( engine_step( &_Engine ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        // TODO: correct error reporting
        return E_FAIL;
    }

    if(_Engine.timer_output_period > 0 && _Engine.time % _Engine.timer_output_period == 0 ) {
        MxPrintPerformanceCounters();
    }

    return S_OK;
}

// TODO: does it make sense to return an hresult???
int MxUniverse_Flag(MxUniverse_Flags flag)
{
    UNIVERSE_CHECKERROR();
    return universe_flags & flag;
}

CAPI_FUNC(HRESULT) MxUniverse_SetFlag(MxUniverse_Flags flag, int value)
{
    UNIVERSE_CHECKERROR();

    if(value) {
        universe_flags |= flag;
    }
    else {
        universe_flags &= ~(flag);
    }

    return MxSimulator::get()->redraw();
}

MxVector3f MxUniverse::getCenter() {
    return engine_center();
}
