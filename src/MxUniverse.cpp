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
#include <MxSimulator.h>
#include <MxUtil.h>
#include <metrics.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <MxThreadPool.hpp>
#include <MxCuboid.hpp>
#include <MxBind.hpp>
#include <state/MxStateVector.h>
#include <state/MxSpeciesList.h>
#include <MxSystem.h>
#include <mx_error.h>
#include <rendering/MxStyle.hpp>
#include <io/MxFIO.h>

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
            mx_exp(std::domain_error(err.c_str())); \
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
    start_step{0},
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

MxUniverse* MxUniverse::get() {
    return &Universe;
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
    std::vector<MxBondHandle*> *bonds = new std::vector<MxBondHandle*>();
    bonds->reserve(_Engine.nr_bonds);

    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        MxBond *b = &_Engine.bonds[i];
        if (b->flags & BOND_ACTIVE)
            bonds->push_back(new MxBondHandle(i));
    }
    return bonds;
    UNIVERSE_FINALLY(NULL);
}

std::vector<MxAngleHandle*> *MxUniverse::angles() {
    UNIVERSE_TRY();
    std::vector<MxAngleHandle*> *angles = new std::vector<MxAngleHandle*>();
    angles->reserve(_Engine.nr_angles);

    for(int i = 0; i < _Engine.nr_angles; ++i) {
        MxAngle *a = &_Engine.angles[i];
        if (a->flags & BOND_ACTIVE)
            angles->push_back(new MxAngleHandle(i));
    }
    return angles;
    UNIVERSE_FINALLY(NULL);
}

std::vector<MxDihedralHandle*> *MxUniverse::dihedrals() {
    UNIVERSE_TRY();
    std::vector<MxDihedralHandle*> *dihedrals = new std::vector<MxDihedralHandle*>();
    dihedrals->reserve(_Engine.nr_dihedrals);

    for(int i = 0; i < _Engine.nr_dihedrals; ++i) {
        MxDihedral *d = &_Engine.dihedrals[i];
        dihedrals->push_back(new MxDihedralHandle(i));
    }
    return dihedrals;
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
    return (MxEventList *)this->events;
}

MxBoundaryConditions *MxUniverse::getBoundaryConditions() {
    return &_Engine.boundary_conditions;
}

double MxUniverse::getKineticEnergy() {
    return engine_kinetic_energy(&_Engine);
}

int MxUniverse::getNumTypes() {
    return _Engine.nr_types;
}

double MxUniverse::getCutoff() {
    return _Engine.s.cutoff;
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

    // Ok to call here, since nothing happens if root element is already released. 
    MxFIO::releaseMxIORootElement();

    // TODO: add support for adaptive time stepping
    // if (dt <= 0.0) dt = _Engine.dt;
    dt = _Engine.dt;

    float dtStore = _Engine.dt;
    _Engine.dt = dt;

    if (until <= 0.0) until = _Engine.dt;

    float tf = _Engine.time + until / dtStore;

    while (_Engine.time < tf) {
        if ( engine_step( &_Engine ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            // TODO: correct error reporting
            return E_FAIL;
        }

        // notify time listeners
        if(Universe.events->eval(_Engine.time * _Engine.dt) != 0) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            // TODO: correct error reporting
            return E_FAIL;
        }

        if(_Engine.timer_output_period > 0 && _Engine.time % _Engine.timer_output_period == 0 ) {
            MxPrintPerformanceCounters();
        }

    }

    _Engine.dt = dtStore;

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


namespace mx { namespace io {

#define MXUNIVERSEIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXUNIVERSEIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxUniverse &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MxUniverse *u = const_cast<MxUniverse*>(&dataElement);

    MXUNIVERSEIOTOEASY(fe, "name", u->name);
    
    MxParticleList pl = *u->particles();
    if(pl.nr_parts > 0) {
        std::vector<MxParticle> particles;
        particles.reserve(pl.nr_parts);
        for(unsigned int i = 0; i < pl.nr_parts; i++) {
            auto ph = pl.item(i);
            if(ph != NULL) {
                auto p = ph->part();
                if(p != NULL && !(p->flags & PARTICLE_NONE))
                    particles.push_back(*p);
            }
        }
        MXUNIVERSEIOTOEASY(fe, "particles", particles);
    }

    // Store bonds; potentials and styles are stored separately to reduce storage
    
    std::vector<MxBondHandle*> bhl = *u->bonds();
    std::vector<MxPotential*> bondPotentials;
    std::vector<std::vector<unsigned int> > bondPotentialIdx;
    std::vector<MxStyle> bondStyles;
    std::vector<MxStyle*> bondStylesP;
    std::vector<std::vector<unsigned int> > bondStyleIdx;
    if(bhl.size() > 0) {
        std::vector<MxBond> bl;
        bl.reserve(bhl.size());
        for(auto bh : bhl) {
            auto b = bh->get();
            if(b->flags & BOND_ACTIVE) {
                if(b->potential != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < bondPotentials.size(); i++) {
                        if(bondPotentials[i] == b->potential) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = bondPotentials.size();
                        bondPotentials.push_back(b->potential);
                        bondPotentialIdx.emplace_back();
                    }
                    bondPotentialIdx[idx].push_back(bl.size());
                }
                if(b->style != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < bondStylesP.size(); i++) {
                        if(bondStylesP[i] == b->style) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = bondStylesP.size();
                        bondStyles.push_back(*b->style);
                        bondStylesP.push_back(b->style);
                        bondStyleIdx.emplace_back();
                    }
                    bondStyleIdx[idx].push_back(bl.size());
                }

                bl.push_back(*b);
            }
        }
        MXUNIVERSEIOTOEASY(fe, "bonds", bl);
        MXUNIVERSEIOTOEASY(fe, "bondPotentials", bondPotentials);
        MXUNIVERSEIOTOEASY(fe, "bondPotentialIdx", bondPotentialIdx);
        MXUNIVERSEIOTOEASY(fe, "bondStyles", bondStyles);
        MXUNIVERSEIOTOEASY(fe, "bondStyleIdx", bondStyleIdx);
    }

    // Store angles; potentials and styles are stored separately to reduce storage
    
    std::vector<MxAngleHandle*> ahl = *u->angles();
    std::vector<MxPotential*> anglePotentials;
    std::vector<std::vector<unsigned int> > anglePotentialIdx;
    std::vector<MxStyle> angleStyles;
    std::vector<MxStyle*> angleStylesP;
    std::vector<std::vector<unsigned int> > angleStyleIdx;
    if(ahl.size() > 0) {
        std::vector<MxAngle> al;
        al.reserve(ahl.size());
        for(auto ah : ahl) {
            auto a = ah->get();
            if(a->flags & ANGLE_ACTIVE) {
                if(a->potential != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < anglePotentials.size(); i++) {
                        if(anglePotentials[i] == a->potential) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = anglePotentials.size();
                        anglePotentials.push_back(a->potential);
                        anglePotentialIdx.emplace_back();
                    }
                    anglePotentialIdx[idx].push_back(al.size());
                }
                if(a->style != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < angleStylesP.size(); i++) {
                        if(angleStylesP[i] == a->style) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = angleStylesP.size();
                        angleStyles.push_back(*a->style);
                        angleStylesP.push_back(a->style);
                        angleStyleIdx.emplace_back();
                    }
                    angleStyleIdx[idx].push_back(al.size());
                }

                al.push_back(*a);
            }
        }
        MXUNIVERSEIOTOEASY(fe, "angles", al);
        MXUNIVERSEIOTOEASY(fe, "anglePotentials", anglePotentials);
        MXUNIVERSEIOTOEASY(fe, "anglePotentialIdx", anglePotentialIdx);
        MXUNIVERSEIOTOEASY(fe, "angleStyles", angleStyles);
        MXUNIVERSEIOTOEASY(fe, "angleStyleIdx", angleStyleIdx);
    }

    // Store dihedrals; potentials and styles are stored separately to reduce storage
    
    std::vector<MxDihedralHandle*> dhl = *u->dihedrals();
    std::vector<MxPotential*> dihedralPotentials;
    std::vector<std::vector<unsigned int> > dihedralPotentialIdx;
    std::vector<MxStyle> dihedralStyles;
    std::vector<MxStyle*> dihedralStylesP;
    std::vector<std::vector<unsigned int> > dihedralStyleIdx;
    if(dhl.size() > 0) {
        std::vector<MxDihedral> dl;
        dl.reserve(dhl.size());
        for(auto dh : dhl) {
            auto d = dh->get();
            if(d != NULL) {
                if(d->potential != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < dihedralPotentials.size(); i++) {
                        if(dihedralPotentials[i] == d->potential) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = dihedralPotentials.size();
                        dihedralPotentials.push_back(d->potential);
                        dihedralPotentialIdx.emplace_back();
                    }
                    dihedralPotentialIdx[idx].push_back(dl.size());
                }
                if(d->style != NULL) {
                    int idx = -1;
                    for(unsigned int i = 0; i < dihedralStylesP.size(); i++) {
                        if(dihedralStylesP[i] == d->style) {
                            idx = i;
                            break;
                        }
                    }

                    if(idx < 0) {
                        idx = dihedralStylesP.size();
                        dihedralStyles.push_back(*d->style);
                        dihedralStylesP.push_back(d->style);
                        dihedralStyleIdx.emplace_back();
                    }
                    dihedralStyleIdx[idx].push_back(dl.size());
                }

                dl.push_back(*d);
            }
        }
        MXUNIVERSEIOTOEASY(fe, "dihedrals", dl);
        MXUNIVERSEIOTOEASY(fe, "dihedralPotentials", dihedralPotentials);
        MXUNIVERSEIOTOEASY(fe, "dihedralPotentialIdx", dihedralPotentialIdx);
        MXUNIVERSEIOTOEASY(fe, "dihedralStyles", dihedralStyles);
        MXUNIVERSEIOTOEASY(fe, "dihedralStyleIdx", dihedralStyleIdx);
    }
    
    MXUNIVERSEIOTOEASY(fe, "temperature", u->getTemperature());
    MXUNIVERSEIOTOEASY(fe, "kineticEnergy", u->getKineticEnergy());

    MxParticleTypeList *ptl = MxParticleTypeList::all();
    std::vector<MxParticleType> partTypes;
    partTypes.reserve(ptl->nr_parts);
    for(unsigned int i = 0; i < ptl->nr_parts; i++) 
        partTypes.push_back(*ptl->item(i));
    MXUNIVERSEIOTOEASY(fe, "particleTypes", partTypes);

    MxPotential *p, *p_cluster;
    std::vector<MxPotential*> pV, pV_cluster;
    std::vector<unsigned int> pIdxA, pIdxB, pIdxA_cluster, pIdxB_cluster;
    for(unsigned int i = 0; i < ptl->nr_parts; i++) {
        for(unsigned int j = i; j < ptl->nr_parts; j++) {
            unsigned int k = ptl->parts[i] * _Engine.max_type + ptl->parts[j];
            p = _Engine.p[k];
            p_cluster = _Engine.p_cluster[k];
            if(p != NULL) {
                pV.push_back(p);
                pIdxA.push_back(i);
                pIdxB.push_back(j);
            }
            if(p_cluster != NULL) {
                pV_cluster.push_back(p_cluster);
                pIdxA_cluster.push_back(i);
                pIdxB_cluster.push_back(j);
            }
        }
    }
    if(pV.size() > 0) {
        MXUNIVERSEIOTOEASY(fe, "potentials", pV);
        MXUNIVERSEIOTOEASY(fe, "potentialTypeA", pIdxA);
        MXUNIVERSEIOTOEASY(fe, "potentialTypeB", pIdxB);
    }
    if(pV_cluster.size() > 0) {
        MXUNIVERSEIOTOEASY(fe, "potentialsCluster", pV_cluster);
        MXUNIVERSEIOTOEASY(fe, "potentialClusterTypeA", pIdxA_cluster);
        MXUNIVERSEIOTOEASY(fe, "potentialClusterTypeB", pIdxB_cluster);
    }

    // save forces
    
    std::vector<MxForce*> forces;
    MxForce *f;
    std::vector<unsigned int> fIdx;
    for(unsigned int i = 0; i < ptl->nr_parts; i++) { 
        auto pTypeId = ptl->parts[i];
        f = _Engine.forces[pTypeId];
        if(f != NULL) {
            bool storeForce = true;
            if(f->isConstant()) {
                MxConstantForce *cf = (MxConstantForce*)f;
                storeForce = cf->userFunc == NULL;
            }
            if(storeForce) {
                forces.push_back(f);
                fIdx.push_back(i);
            }
        }
    }
    if(forces.size() > 0) {
        MXUNIVERSEIOTOEASY(fe, "forces", forces);
        MXUNIVERSEIOTOEASY(fe, "forceType", fIdx);
    }

    delete ptl;

    fileElement->type = "Universe";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxUniverse *dataElement) {

    MxIOChildMap::const_iterator feItr;

    MxFIO::importSummary = new MxFIOImportSummary();

    MxUniverse *universe = MxUniverse::get();

    MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "name", &universe->name);
    MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "temperature", &_Engine.temperature);

    // Setup data should have already been intercepted by this stage, so populate universe

    // load types
    //      Special handling here: export includes default types, 
    //      which must be skipped here to avoid duplicating imported defaults with those of installation

    feItr = fileElement.children.find("particleTypes");
    MxIOElement *fePartTypes = feItr->second;
    for(unsigned int i = 0; i < fePartTypes->children.size(); i++) {
        if(i < 2) { 
            MxFIO::importSummary->particleTypeIdMap[i] = i;
        } 
        else {
            MxIOElement *fePartType = fePartTypes->children[std::to_string(i)];
            MxParticleType partType;
            auto typeId = partType.id;
            mx::io::fromFile(*fePartType, metaData, &_Engine.types[typeId]);
            MxFIO::importSummary->particleTypeIdMap[i] = typeId;
        }
    }

    // load potentials

    if(fileElement.children.find("potentials") != fileElement.children.end()) {
        std::vector<MxPotential*> pV;
        std::vector<unsigned int> pIdxA, pIdxB;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentials", &pV);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypeA", &pIdxA);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypeB", &pIdxB);
        for(unsigned int i = 0; i < pV.size(); i++) {
            auto typeIdA = MxFIO::importSummary->particleTypeIdMap[pIdxA[i]];
            auto typeIdB = MxFIO::importSummary->particleTypeIdMap[pIdxB[i]];
            MxBind::types(pV[i], &_Engine.types[typeIdA], &_Engine.types[typeIdB]);
        }
    }
    if(fileElement.children.find("potentialsCluster") != fileElement.children.end()) {
        std::vector<MxPotential*> pV_cluster;
        std::vector<unsigned int> pIdxA_cluster, pIdxB_cluster;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentialsCluster", &pV_cluster);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentialClusterTypeA", &pIdxA_cluster);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "potentialClusterTypeB", &pIdxB_cluster);
        for(unsigned int i = 0; i < pV_cluster.size(); i++) {
            auto typeIdA = MxFIO::importSummary->particleTypeIdMap[pIdxA_cluster[i]];
            auto typeIdB = MxFIO::importSummary->particleTypeIdMap[pIdxB_cluster[i]];
            MxBind::types(pV_cluster[i], &_Engine.types[typeIdA], &_Engine.types[typeIdB], true);
        }
    }

    // load forces

    if(fileElement.children.find("forces") != fileElement.children.end()) {
        std::vector<MxForce*> forces;
        std::vector<unsigned int> fIdx;
        int fSVIdx;
        
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "forces", &forces);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "forceType", &fIdx);
        for(unsigned int i = 0; i < fIdx.size(); i++) { 
            auto pType = &_Engine.types[MxFIO::importSummary->particleTypeIdMap[fIdx[i]]];
            MxForce *f = forces[i];
            MxBind::force(f, pType);
        }
    }
    
    // load particles

    if(fileElement.children.find("particles") != fileElement.children.end()) {

        std::vector<MxParticle> particles;
        MxParticle *part;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "particles", &particles);
        auto feParticles = feItr->second;
        for(unsigned int i = 0; i < particles.size(); i++) {
            MxParticle &p = particles[i];
            auto typeId = MxFIO::importSummary->particleTypeIdMap[p.typeId];
            auto pType = &_Engine.types[typeId];
            int32_t clusterId = p.clusterId > 0 ? MxFIO::importSummary->particleIdMap[p.clusterId] : p.clusterId;
            auto ph = MxParticle_New(pType, &p.position, &p.velocity, &clusterId);
            auto part = ph->part();
            auto pId = p.id;
            MxFIO::importSummary->particleIdMap[pId] = part->id;

            part->radius = p.radius;
            part->mass = p.mass;
            part->imass = p.imass;
            part->flags = p.flags;
            part->creation_time = p.creation_time;
            if(p.state_vector) {
                for(unsigned int j = 0; j < p.state_vector->size; j++) {
                    MxSpecies *species = p.state_vector->species->item(j);
                    int k = part->state_vector->species->index_of(species->getId().c_str());
                    if(k >= 0) 
                        part->state_vector->fvec[k] = p.state_vector->fvec[j];
                }
            }
            if(p.style) 
                part->style = new MxStyle(*p.style);
        }

    }

    // load bonds; potentials and styles are stored separately to reduce storage

    if(fileElement.children.find("bonds") != fileElement.children.end()) {
        std::vector<MxBond> bonds;
        std::vector<MxPotential*> bondPotentials;
        std::vector<std::vector<unsigned int> > bondPotentialIdx;
        std::vector<MxStyle> bondStyles;
        std::vector<std::vector<unsigned int> > bondStyleIdx;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "bonds", &bonds);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "bondPotentials", &bondPotentials);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "bondPotentialIdx", &bondPotentialIdx);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "bondStyles", &bondStyles);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "bondStyleIdx", &bondStyleIdx);
        std::vector<MxBondHandle> bondsCreated(bonds.size(), MxBondHandle());

        for(unsigned int i = 0; i < bondPotentialIdx.size(); i++) { 
            auto bIndices = bondPotentialIdx[i];
            MxPotential *p = bondPotentials[i];
            for(auto bIdx : bIndices) {
                auto b = bonds[bIdx];
                MxBondHandle bh(
                    p, 
                    MxFIO::importSummary->particleIdMap[b.i], 
                    MxFIO::importSummary->particleIdMap[b.j], 
                    b.half_life, b.dissociation_energy, b.flags
                );
                auto be = bh.get();

                be->creation_time = b.creation_time;
                bondsCreated[bIdx] = bh;
            }
        }

        for(unsigned int i = 0; i < bondStyleIdx.size(); i++) {
            auto bIndices = bondStyleIdx[i];
            MxStyle *s = new MxStyle(bondStyles[i]);
            for(auto bIdx : bIndices) {
                auto bh = bondsCreated[bIdx];
                if(bh.id >= 0) 
                    bh.get()->style = s;
            }
        }
    }

    // load angles; potentials and styles are stored separately to reduce storage
    
    if(fileElement.children.find("angles") != fileElement.children.end()) {
        std::vector<MxAngle> angles;
        std::vector<MxPotential*> anglePotentials;
        std::vector<std::vector<unsigned int> > anglePotentialIdx;
        std::vector<MxStyle> angleStyles;
        std::vector<std::vector<unsigned int> > angleStyleIdx;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "angles", &angles);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "anglePotentials", &anglePotentials);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "anglePotentialIdx", &anglePotentialIdx);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "angleStyles", &angleStyles);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "angleStyleIdx", &angleStyleIdx);
        std::vector<MxAngleHandle*> anglesCreated(angles.size(), 0);

        for(unsigned int i = 0; i < anglePotentialIdx.size(); i++) { 
            auto aIndices = anglePotentialIdx[i];
            MxPotential *p = anglePotentials[i];
            for(auto aIdx : aIndices) {
                auto a = angles[aIdx];
                MxParticle *pi, *pj, *pk;
                pi = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[a.i]];
                pj = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[a.j]];
                pk = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[a.k]];
                auto ah = MxAngle::create(p, pi->py_particle(), pj->py_particle(), pk->py_particle(), a.flags);
                auto ae = ah->get();

                ae->half_life = a.half_life;
                ae->dissociation_energy = a.dissociation_energy;
                ae->creation_time = a.creation_time;
                anglesCreated[aIdx] = ah;
            }
        }

        for(unsigned int i = 0; i < angleStyleIdx.size(); i++) {
            auto aIndices = angleStyleIdx[i];
            MxStyle *s = new MxStyle(angleStyles[i]);
            for(auto aIdx : aIndices) {
                auto a = anglesCreated[aIdx];
                if(a != NULL) 
                    a->get()->style = s;
            }
        }
    }

    // load dihedrals; potentials and styles are stored separately to reduce storage
    
    if(fileElement.children.find("dihedrals") != fileElement.children.end()) {
        std::vector<MxDihedral> dihedrals;
        std::vector<MxPotential*> dihedralPotentials;
        std::vector<std::vector<unsigned int> > dihedralPotentialIdx;
        std::vector<MxStyle> dihedralStyles;
        std::vector<std::vector<unsigned int> > dihedralStyleIdx;
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "dihedrals", &dihedrals);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "dihedralPotentials", &dihedralPotentials);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "dihedralPotentialIdx", &dihedralPotentialIdx);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "dihedralStyles", &dihedralStyles);
        MXUNIVERSEIOFROMEASY(feItr, fileElement.children, metaData, "dihedralStyleIdx", &dihedralStyleIdx);
        std::vector<MxDihedralHandle*> dihedralsCreated(dihedrals.size(), 0);

        for(unsigned int i = 0; i < dihedralPotentialIdx.size(); i++) { 
            auto dIndices = dihedralPotentialIdx[i];
            MxPotential *p = dihedralPotentials[i];
            for(auto dIdx : dIndices) {
                auto d = dihedrals[dIdx];
                MxParticle *pi, *pj, *pk, *pl;
                pi = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[d.i]];
                pj = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[d.j]];
                pk = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[d.k]];
                pl = _Engine.s.partlist[MxFIO::importSummary->particleIdMap[d.l]];
                auto dh = MxDihedral::create(p, pi->py_particle(), pj->py_particle(), pk->py_particle(), pl->py_particle());
                auto de = dh->get();

                de->half_life = d.half_life;
                de->dissociation_energy = d.dissociation_energy;
                de->creation_time = d.creation_time;
                dihedralsCreated[dIdx] = dh;
            }
        }

        for(unsigned int i = 0; i < dihedralStyleIdx.size(); i++) {
            auto dIndices = dihedralStyleIdx[i];
            MxStyle *s = new MxStyle(dihedralStyles[i]);
            for(auto dIdx : dIndices) {
                auto d = dihedralsCreated[dIdx];
                if(d != NULL) 
                    d->get()->style = s;
            }
        }
    }

    return S_OK;
}

}};
