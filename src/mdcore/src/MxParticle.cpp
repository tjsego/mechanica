/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/


/* include some standard header files */
#define _USE_MATH_DEFINES // for C++

#include <MxParticle.h>
#include "engine.h"

#include "space.h"
#include "mx_runtime.h"

#include "../../rendering/MxStyle.hpp"
#include "MxCluster.hpp"
#include "metrics.h"
#include "MxParticleList.hpp"
#include "../../MxUtil.h"
#include "../../MxLogger.h"
#include "../../mx_error.h"
#include "../../io/MxFIO.h"

#include <../../state/MxSpeciesList.h>
#include <../../state/MxStateVector.h>

#include <sstream>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include "fptype.h"
#include <iostream>
#include <unordered_map>
#include <typeinfo>

MxParticle::MxParticle() {
    bzero(this, sizeof(MxParticle));
}

struct Foo {
    int x; int y; int z;
};

static unsigned colors [] = {
    0xCCCCCC,
    0x6D99D3, // Rust Oleum Spa Blue
    0xF65917, // Rust Oleum Pumpkin
    0xF9CB20, // rust oleum yellow
    0x3CB371, // green
    0x6353BB, // SGI purple
    0xf4AC21, // gold
    0xC4A5DF, // light purple
    0xDC143C, // dark red
    0x1E90FF, // blue
    0xFFFF00, // yellow
    0x8A2BE2, // purple
    0x76D7C4, // light green
    0xF08080, // salmon
    0xFF00FF, // fuscia
    0xFF8C00, // orange
    0xFFDAB9, // tan / peach
    0x7F8C8D, // gray
    0x884EA0, // purple
    0x6B8E23,
    0x00FFFF,
    0xAFEEEE,
    0x008080,
    0xB0E0E6,
    0x6495ED,
    0x191970,
    0x0000CD,
    0xD8BFD8,
    0xFF1493,
    0xF0FFF0,
    0xFFFFF0,
    0xFFE4E1,
    0xDCDCDC,
    0x778899,
    0x000000
};

unsigned int *MxParticle_Colors = colors;

#define PARTICLE_SELF(pypart) \
    MxParticle *self = pypart->part(); \
    if(self == NULL) { \
        throw std::runtime_error("Particle has been destroyed or is invalid"); \
        return NULL; \
    }

#define PARTICLE_PROP_SELF(pypart) \
    MxParticle *self = pypart->part(); \
    if(self == NULL) { \
        throw std::runtime_error("Particle has been destroyed or is invalid"); \
        return -1; \
    }

#define PARTICLE_SELFW(pypart, ret) \
    MxParticle *self = pypart->part(); \
    if(self == NULL) { \
        Log(LOG_WARNING) << "Particle has been destroyed or is invalid"; \
        return ret; \
    }

#define PARTICLE_TYPE(pypart) \
    MxParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        throw std::runtime_error("Particle has been destroyed or is invalid"); \
        return NULL; \
    }

#define PARTICLE_PROP_TYPE(pypart) \
    MxParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        throw std::runtime_error("Particle has been destroyed or is invalid"); \
        return -1; \
    }

#define PARTICLE_TYPEW(pypart, ret) \
    MxParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        Log(LOG_WARNING) << "Particle has been destroyed or is invalid"; \
        return ret; \
    }

static int particle_init(MxParticleHandle *self, MxVector3f *position=NULL, MxVector3f *velocity=NULL, int *cluster=NULL);

static int particle_init_ex(MxParticleHandle *self,  const MxVector3f &position,
                            const MxVector3f &velocity,
                            int clusterId);


static MxParticleList *particletype_items(MxParticleType *self);

static MxParticle *particleSelf(MxParticleHandle *handle);


struct Offset {
    uint32_t kind;
    uint32_t offset;
};

static_assert(sizeof(Offset) == sizeof(void*), "error, void* must be 64 bit");

double MxParticleHandle::getCharge() {
    PARTICLE_SELFW(this, 0)
    return self->q;
}

void MxParticleHandle::setCharge(const double &charge) {
    PARTICLE_SELFW(this, )
    self->q = charge;
}

double MxParticleHandle::getMass() {
    PARTICLE_SELFW(this, 0)
    return self->mass;
}

void MxParticleHandle::setMass(const double &mass) {
    if(mass <= 0.f) {
        mx_error(E_FAIL, "Mass must be positive");
        return;
    }

    PARTICLE_SELFW(this, )
    self->mass = mass;
    self->imass = 1.f / mass;
}

bool MxParticleType::getFrozen() {
    return particle_flags & PARTICLE_FROZEN;
}

void MxParticleType::setFrozen(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN;
    else particle_flags &= ~PARTICLE_FROZEN;
}

bool MxParticleType::getFrozenX() {
    return particle_flags & PARTICLE_FROZEN;
}

void MxParticleType::setFrozenX(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_X;
    else particle_flags &= ~PARTICLE_FROZEN_X;
}

bool MxParticleType::getFrozenY() {
    return particle_flags & PARTICLE_FROZEN_Y;
}

void MxParticleType::setFrozenY(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_Y;
    else particle_flags &= ~PARTICLE_FROZEN_Y;
}

bool MxParticleType::getFrozenZ() {
    return particle_flags & PARTICLE_FROZEN_Z;
}

void MxParticleType::setFrozenZ(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_Z;
    else particle_flags &= ~PARTICLE_FROZEN_Z;
}

bool MxParticleHandle::getFrozen() {
    PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN;
}

void MxParticleHandle::setFrozen(const bool frozen) {
    PARTICLE_SELFW(this, )
    if(frozen) self->flags |= PARTICLE_FROZEN;
    else self->flags &= ~PARTICLE_FROZEN;
}

bool MxParticleHandle::getFrozenX() {
    PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_X;
}

void MxParticleHandle::setFrozenX(const bool frozen) {
    PARTICLE_SELFW(this, )
    if(frozen) self->flags |= PARTICLE_FROZEN_X;
    else self->flags &= ~PARTICLE_FROZEN_X;
}

bool MxParticleHandle::getFrozenY() {
    PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_Y;
}

void MxParticleHandle::setFrozenY(const bool frozen) {
    PARTICLE_SELFW(this, )
    if(frozen) self->flags |= PARTICLE_FROZEN_Y;
    else self->flags &= ~PARTICLE_FROZEN_Y;
}

bool MxParticleHandle::getFrozenZ() {
    PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_Z;
}

void MxParticleHandle::setFrozenZ(const bool frozen) {
    PARTICLE_SELFW(this, )
    if(frozen) self->flags |= PARTICLE_FROZEN_Z;
    else self->flags &= ~PARTICLE_FROZEN_Z;
}

MxStyle *MxParticleHandle::getStyle() {
    PARTICLE_SELFW(this, 0)
    return self->style;
}

void MxParticleHandle::setStyle(MxStyle *style) {
    PARTICLE_SELFW(this, )
    self->style = style;
}

double MxParticleHandle::getAge() {
    PARTICLE_SELFW(this, 0)
    return (_Engine.time - self->creation_time) * _Engine.dt;
}

double MxParticleHandle::getRadius() {
    PARTICLE_SELFW(this, 0)
    return self->radius;
}

void MxParticleHandle::setRadius(const double &radius) {
    PARTICLE_SELFW(this, )
    self->radius = radius;
}

std::string MxParticleHandle::getName() {
    PARTICLE_TYPEW(this, "")
    return ptype->name;
}

std::string MxParticleHandle::getName2() {
    PARTICLE_TYPEW(this, "")
    return ptype->name2;
}

double MxParticleType::getTemperature() {
    return this->kinetic_energy;
}

double MxParticleType::getTargetTemperature() {
    return this->target_energy;
}

void MxParticleType::setTargetTemperature(const double &temperature) {
    this->target_energy = temperature;
}

MxVector3f MxParticleHandle::getPosition() {
    PARTICLE_SELFW(this, MxVector3f(0.0))
    return self->global_position();
}

void MxParticleHandle::setPosition(MxVector3f position) {
    PARTICLE_SELFW(this, )
    self->set_global_position(position);
}

MxVector3f &MxParticleHandle::getVelocity() {
    auto self = particleSelf(this);
    return self->velocity;
}

void MxParticleHandle::setVelocity(MxVector3f velocity) {
    PARTICLE_SELFW(this, )
    self->velocity = velocity;
}

MxVector3f MxParticleHandle::getForce() {
    PARTICLE_SELFW(this, MxVector3f(0.0))
    return self->force;
}

MxVector3f &MxParticleHandle::getForceInit() {
    auto self = particleSelf(this);
    return self->force_i;
}

void MxParticleHandle::setForceInit(MxVector3f force) {
    PARTICLE_SELFW(this, )
    self->force_i = force;
}

int MxParticleHandle::getId() {
    PARTICLE_SELFW(this, 0)
    return self->id;
}

int16_t MxParticleHandle::getTypeId() {
    PARTICLE_SELFW(this, 0)
    return self->typeId;
}

int32_t MxParticleHandle::getClusterId() {
    PARTICLE_SELFW(this, -1)
    return self->clusterId;
}

uint16_t MxParticleHandle::getFlags() {
    PARTICLE_SELFW(this, 0)
    return self->flags;
}

MxStateVector *MxParticleHandle::getSpecies() {
    PARTICLE_SELFW(this, 0)
    return self->state_vector;
}

MxParticleHandle::operator MxClusterParticleHandle*() {
    PARTICLE_TYPEW(this, NULL)
    if (!ptype->isCluster()) return NULL;
    return static_cast<MxClusterParticleHandle*>(this);
}

int typeIdByName(const char *_name) {
    for(int i = 0; i < _Engine.nr_types; i++) {
        if(strcmp(_Engine.types[i].name, _name) == 0)
            return _Engine.types[i].id;
    }

    return -1;
}

// Check whether a type name has been registered
bool checkTypeName(const char *_name) {
    return typeIdByName(_name) >= 0;
}

// Check whether a type name has been registered among all registered derived types
bool checkDerivedTypeName(const char *_name) {
    return typeIdByName(_name) >= 2;
}

// If the returned type id is the same as the two base types, 
// then generate a name according to type info. 
// Otherwise, this type presumably has already been registered with a unique name. 
std::string getUniqueName(MxParticleType *type) {
    if(checkTypeName(type->name) && !checkDerivedTypeName(type->name)) return typeid(*type).name();
    return type->name;
}

void assignUniqueTypeName(MxParticleType *type) {
    if(checkDerivedTypeName(type->name)) return;

    std::strncpy(type->name, getUniqueName(type).c_str(), MxParticleType::MAX_NAME);
}

HRESULT MxParticleType_checkRegistered(MxParticleType *type) {
    if(!type) return 0;
    int typeId = typeIdByName(type->name);
    return typeId > 1;
}

HRESULT _MxParticle_init()
{
    if(engine::max_type < 3) {
        return mx_error(E_FAIL, "must have at least space for 3 particle types");
    }

    if(engine::nr_types != 0) {
        return mx_error(E_FAIL, "engine types already set");
    }

    if((engine::types = (MxParticleType *)malloc( sizeof(MxParticleType) * engine::max_type )) == NULL ) {
        return mx_error(E_FAIL, "could not allocate types memory");
    }
    
    ::memset(engine::types, 0, sizeof(MxParticleType) * engine::max_type);

    engine::nr_types = 0;
    auto type = new MxParticleType();
    
    return S_OK;
}

MxParticleType* MxParticleType_New(const char *_name) {
    auto type = new MxParticleType(*MxParticle_GetType());
    std::strncpy(type->name, std::string(_name).c_str(), MxParticleType::MAX_NAME);
    return type;
}

MxParticleType* MxParticleType_ForEngine(struct engine *e, double mass,
        double charge, const char *name, const char *name2)
{
    MxParticleType *result = MxParticleType_New(name);
    
    result->mass = mass;
    result->charge = charge;
    std::strncpy(result->name2, std::string(name2).c_str(), MxParticleType::MAX_NAME);

    return result;
}

MxParticleType::MxParticleType(const bool &noReg) {
    radius = 1.0;
    minimum_radius = 0.0;
    mass = 1.0;
    charge = 0.0;
    id = 0;
    dynamics = PARTICLE_NEWTONIAN;
    type_flags = PARTICLE_TYPE_NONE;
    particle_flags = PARTICLE_NONE;
    
    auto c = Magnum::Color3::fromSrgb(colors[(_Engine.nr_types - 1) % (sizeof(colors)/sizeof(unsigned))]);
    style = new MxStyle(&c);

    ::strncpy(name, "Particle", MxParticleType::MAX_NAME);
    ::strncpy(name2, "Particle", MxParticleType::MAX_NAME);

    if(!noReg) registerType();
}

CAPI_FUNC(MxParticleType*) MxParticle_GetType()
{
    return &engine::types[0];
}

CAPI_FUNC(MxParticleType*) MxCluster_GetType()
{
    return &engine::types[1];
}

MxParticle *particleSelf(MxParticleHandle *handle) {
    PARTICLE_SELF(handle)
    return self;
}

HRESULT MxParticleHandle::destroy()
{
    PARTICLE_SELFW(this, S_OK)
    return engine_del_particle(&_Engine, self->id);
}

MxVector3f MxParticleHandle::sphericalPosition(MxParticle *particle, MxVector3f *origin)
{
    MxVector3f _origin;

    if (particle) _origin = particle->global_position();
    else if (origin) _origin = *origin;
    else _origin = engine_center();
    return MxCartesianToSpherical(part()->global_position(), _origin);
}

MxVector3f MxParticleHandle::relativePosition(const MxVector3f &origin, const bool &comp_bc) {
    return MxRelativePosition(this->getPosition(), origin, comp_bc);
}

MxMatrix3f MxParticleHandle::virial(float *radius)
{
    PARTICLE_SELFW(this, MxMatrix3f(0.0))

    MxVector3f pos = self->global_position();
    MxMatrix3f mat;
    
    float _radius = radius ? *radius : self->radius * 10;
    
    std::set<short int> typeIds;
    for(int i = 0; i < _Engine.nr_types; ++i) {
        typeIds.emplace(i);
    }
    
    MxCalculateVirial(pos.data(), _radius, typeIds, mat.data());

    return mat;
}

HRESULT MxParticleType::addpart(int32_t id)
{
    this->parts.insert(id);
    return S_OK;
}


/**
 * remove a particle id from this type
 */
HRESULT MxParticleType::del_part(int32_t id) {
    this->parts.remove(id);
    return S_OK;
}

std::set<short int> MxParticleType::particleTypeIds() {
	std::set<short int> ids;

	for(int i = 0; i < _Engine.nr_types; ++i) {
		if(!_Engine.types[i].isCluster()) ids.insert(i);
	}

	return ids;
}

bool MxParticleType::isCluster() {
    return this->particle_flags & PARTICLE_CLUSTER;
}

MxParticleType::operator MxClusterParticleType*() {
    if (!this->isCluster()) return NULL;
    return static_cast<MxClusterParticleType*>(this);
}

MxParticleHandle *MxParticleType::operator()(MxVector3f *position,
                                             MxVector3f *velocity,
                                             int *clusterId) 
{
    return MxParticle_New(this, position, velocity, clusterId);
}

MxParticleHandle *MxParticleType::operator()(const std::string &str, int *clusterId) {
    MxParticle *dummy = MxParticle::fromString(str);

    MxParticleHandle *ph = (*this)(&dummy->position, &dummy->velocity, clusterId);
    auto p = ph->part();

    // copy reamining valid imported data

    p->force = dummy->force;
    p->force_i = dummy->force_i;
    p->inv_number_density = dummy->inv_number_density;
    p->creation_time = dummy->creation_time;
    p->persistent_force = dummy->persistent_force;
    p->radius = dummy->radius;
    p->mass = dummy->mass;
    p->imass = dummy->imass;
    p->q = dummy->q;
    p->p0 = dummy->p0;
    p->v0 = dummy->v0;
    for(unsigned int i = 0; i < 4; i++) {
        p->xk[i] = dummy->xk[i];
        p->vk[i] = dummy->vk[i];
    }
    p->vid = dummy->vid;
    p->flags = dummy->flags;

    delete dummy;

    return ph;
}

std::vector<int> MxParticleType::factory(unsigned int nr_parts, 
                                         std::vector<MxVector3f> *positions, 
                                         std::vector<MxVector3f> *velocities, 
                                         std::vector<int> *clusterIds) 
{
    return MxParticles_New(this, nr_parts, positions, velocities, clusterIds);
}

MxParticleType* MxParticleType::newType(const char *_name) {
    auto type = new MxParticleType(*this);
    std::strncpy(type->name, std::string(_name).c_str(), MxParticleType::MAX_NAME);
    return type;
}

HRESULT MxParticleType::registerType() {
    if (isRegistered()) return S_OK;

    if(engine::nr_types >= engine::max_type) {
        mx_exp(std::runtime_error("out of memory for new particle type"));
        return E_OUTOFMEMORY;
    }

    if(engine::nr_types >= 2 && !checkDerivedTypeName(this->name)) {
        assignUniqueTypeName(this);
        Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << this->name;
    }

    Log(LOG_DEBUG) << "Creating new particle type " << engine::nr_types;

    if(this->mass > 0.f) this->imass = 1.0f / this->mass;
    
    this->id = engine::nr_types;
    memcpy(&engine::types[engine::nr_types], this, sizeof(MxParticleType));
    engine::nr_types++;

    // invoke callbacks
    this->on_register();

    return S_OK;
}

bool MxParticleType::isRegistered() { return MxParticleType_checkRegistered(this); }

MxParticleType *MxParticleType::get() {
    return MxParticleType_FindFromName(this->name);
}

MxParticleHandle *MxParticle_FissionSimple(MxParticle *self,
        MxParticleType *a, MxParticleType *b, int nPartitionRatios,
        float *partitionRations)
{
    Log(LOG_TRACE) << "Executing simple fission " << self->id << ", " << (int)self->typeId;

    int self_id = self->id;

    MxParticleType *type = &_Engine.types[self->typeId];
    
    // volume preserving radius
    float r2 = self->radius / std::pow(2., 1/3.);
    
    if(r2 < type->minimum_radius) {
        return NULL;
    }

    MxParticle part = {};
    part.mass = self->mass;
    part.position = self->position;
    part.velocity = self->velocity;
    part.force = {};
    part.persistent_force = {};
    part.q = self->q;
    part.radius = self->radius;
    part.id = engine_next_partid(&_Engine);
    part.vid = 0;
    part.typeId = type->id;
    part.flags = self->flags;
    part._pyparticle = NULL;
    part.parts = NULL;
    part.nr_parts = 0;
    part.size_parts = 0;
    part.creation_time = _Engine.time;
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }

    std::uniform_real_distribution<float> x(-1, 1);

    MxRandomType &MxRandom = MxRandomEngine();
    MxVector3f sep = {x(MxRandom), x(MxRandom), x(MxRandom)};
    sep = sep.normalized();
    sep = sep * r2;

    try {

        // create a new particle at the same location as the original particle.
        MxParticle *p = NULL;
        MxVector3f vec(0.0);
        int result = space_getpos(&_Engine.s, self->id, vec.data());

        if(result < 0) {
            Log(LOG_CRITICAL) << part.typeId << ", " << _Engine.nr_types;
            Log(LOG_CRITICAL) << vec;
            Log(LOG_CRITICAL) << part.id << ", " << self->id << ", " << _Engine.s.nr_parts;
            mx_exp(std::runtime_error(engine_err_msg[-engine_err]));
            return NULL;            
        }

        // Double-check boundaries
        const MxBoundaryConditions &bc = _Engine.boundary_conditions;
        std::vector<bool> periodicFlags {
            bool(bc.periodic & space_periodic_x), 
            bool(bc.periodic & space_periodic_y), 
            bool(bc.periodic & space_periodic_z)
        };

        // Calculate new positions; account for boundaries
        MxVector3f posParent = vec + sep;
        MxVector3d posChild = MxVector3d(vec - sep);

        for(unsigned int i = 0; i < 3; i++) {
            double dim_i = _Engine.s.dim[i];
            double origin_i = _Engine.s.origin[i];

            if(periodicFlags[i]) {
                while(posChild[i] >= dim_i + origin_i) 
                    posChild[i] -= dim_i;

                while(posChild[i] < origin_i) 
                    posChild[i] += dim_i;
            }
        }

        result = engine_addpart(&_Engine, &part, posChild.data(), &p);

        if(result < 0) {
            Log(LOG_CRITICAL) << part.typeId << ", " << _Engine.nr_types;
            Log(LOG_CRITICAL) << posParent;
            Log(LOG_CRITICAL) << posChild;
            Log(LOG_CRITICAL) << part.id << ", " << _Engine.s.nr_parts;
            mx_exp(std::runtime_error(engine_err_msg[-engine_err]));
            return NULL;
        }
        
        // pointers after engine_addpart could change...
        self = _Engine.s.partlist[self_id];
        space_setpos(&_Engine.s, self_id, posParent.data());
        Log(LOG_DEBUG) << self->position << ", " << p->position;
        
        // all is good, set the new radii
        self->radius = r2;
        p->radius = r2;
        self->mass = p->mass = self->mass / 2.;
        self->imass = p->imass = 1. / self->mass;

        Log(LOG_TRACE) << "Simple fission for type " << (int)_Engine.types[self->typeId].id;

        return p->py_particle();
    }
    catch (const std::exception &e) {
        Log(LOG_ERROR);
        mx_exp(e); return NULL;
    }
}

MxParticleHandle* MxParticleHandle::fission()
{
    try {
        PARTICLE_SELF(this)
        return MxParticle_FissionSimple(self, NULL, NULL, 0, NULL);
    }
    catch (const std::exception &e) {
        mx_exp(e); return NULL;
    }
}

MxParticleHandle *MxParticleHandle::split() { return fission(); }

MxParticle* MxParticle_Get(MxParticleHandle *pypart) {
    return _Engine.s.partlist[pypart->id];
}

MxParticleHandle *MxParticle::py_particle() {
    
    if(!this->_pyparticle) this->_pyparticle = new MxParticleHandle(this->id, this->typeId);
    
    return this->_pyparticle;
}


HRESULT MxParticle::addpart(int32_t pid) {

    // only in clusters
    if (!_Engine.types[typeId].isCluster()) return mx_error(E_FAIL, "not a cluster");
    
    /* do we need to extend the partlist? */
    if ( nr_parts == size_parts ) {
        size_parts += CLUSTER_PARTLIST_INCR;
        int32_t* temp;
        if ( ( temp = (int32_t*)malloc( sizeof(int32_t) * size_parts ) ) == NULL )
            return mx_error(E_FAIL, "could not allocate space for type particles");
        memcpy( temp , parts , sizeof(int32_t) * nr_parts );
        free( parts );
        parts = temp;
    }
    
    MxParticle *p = _Engine.s.partlist[pid];
    p->clusterId = this->id;
    
    parts[nr_parts] = pid;
    nr_parts++;
    return S_OK;
}

HRESULT MxParticle::removepart(int32_t pid) {
    
    int pid_index = -1;
    
    for(int i = 0; i < this->nr_parts; ++i) {
        if(this->particle(i)->id == pid) {
            pid_index = i;
            break;
        }
    }
    
    if(pid_index < 0) {
        return mx_error(E_FAIL, "particle id not in this cluster");
    }
    
    MxParticle *p = _Engine.s.partlist[pid];
    p->clusterId = -1;
    
    for(int i = pid_index; i + 1 < this->nr_parts; ++i) {
        this->parts[i] = this->parts[i+1];
    }
    nr_parts--;
    
    return S_OK;
}

bool MxParticle::verify() {
    bool gte, lt;
    
    if(this->flags & PARTICLE_LARGE) {
        gte = x[0] >= 0 && x[1] >= 0 && x[2] >= 0;
        auto eng_dims = engine_dimensions();
        lt = x[0] <= eng_dims[0] && x[1] <= eng_dims[1] &&x[2] <= eng_dims[2];
    }
    else {
        gte = x[0] >= 0 && x[1] >= 0 && x[2] >= 0;
        // TODO, make less than
        lt = x[0] <= _Engine.s.h[0] && x[1] <= _Engine.s.h[1] &&x[2] <= _Engine.s.h[2];
    }
    
    bool pindex = this == _Engine.s.partlist[this->id];

    if(!gte || !lt || !pindex) {
        
        Log(LOG_ERROR) << "Verify failed for particle " << this->id;
        Log(LOG_ERROR) << "   Large particle   : " << (this->flags & PARTICLE_LARGE);
        Log(LOG_ERROR) << "   Validated lower  : " << lt;
        Log(LOG_ERROR) << "   Validated upper  : " << gte;
        Log(LOG_ERROR) << "   Validated address: " << pindex;
        Log(LOG_ERROR) << "   Global position  : " << this->global_position();
        Log(LOG_ERROR) << "   Relative position: " << this->position;
        Log(LOG_ERROR) << "   Particle type    : " << this->py_particle()->type()->name;
        Log(LOG_ERROR) << "   Engine dims      : " << engine_dimensions();

        if(!(this->flags & PARTICLE_LARGE)) {
            space_cell *cell = _Engine.s.celllist[this->id];
            Log(LOG_ERROR) << "   Cell dims        : " << MxVector3d::from(_Engine.s.h);
            Log(LOG_ERROR) << "   Cell location    : " << MxVector3i::from(cell->loc);
            Log(LOG_ERROR) << "   Cell origin      : " << MxVector3d::from(cell->origin);
        }
    }

    assert("particle pos below zero" && gte);
    assert("particle pos over cell size" && lt);
    assert("particle not in correct partlist location" && pindex);
    return gte && lt && pindex;
}

MxParticle::operator MxCluster*() {
    MxParticleType *type = &_Engine.types[typeId];
    if (!type->isCluster()) return NULL;
    return static_cast<MxCluster*>(static_cast<void*>(this));
}

MxParticleHandle* MxParticle_New(MxParticleType *type, 
                                 MxVector3f *position,
                                 MxVector3f *velocity,
                                 int *clusterId) 
{
    
    if(!type) {
        return NULL;
    }
    
    // make a new pyparticle
    auto pyPart = new MxParticleHandle();
    pyPart->typeId = type->id;
    
    if(particle_init(pyPart, position, velocity, clusterId) < 0) {
        Log(LOG_ERROR) << "failed calling particle_init";
        return NULL;
    }
    
    return pyPart;
}

std::vector<int> MxParticles_New(std::vector<MxParticleType*> types, 
                                 std::vector<MxVector3f> *positions, 
                                 std::vector<MxVector3f> *velocities, 
                                 std::vector<int> *clusterIds) 
{
    MxVector3f *position = NULL, *velocity = NULL;
    int *clusterId = NULL;

    unsigned int nr_parts = types.size();

    if((positions && positions->size() != nr_parts) || (velocities && velocities->size() != nr_parts) || (clusterIds && clusterIds->size() != nr_parts)) {
        mx_exp(std::runtime_error("Incosistent element inputs"));
        return {};
    }

    if(_Engine.s.nr_parts + nr_parts > _Engine.s.size_parts) { 
        int size_incr = (int((_Engine.s.nr_parts - _Engine.s.size_parts + nr_parts) / space_partlist_incr) + 1) * space_partlist_incr;
        if(space_growparts(&_Engine.s, size_incr) != space_err_ok) { 
            mx_exp(std::runtime_error("failed calling space_growparts"));
            return {};
        }
    }

    std::vector<int> result(nr_parts, -1);
    for(unsigned int i = 0; i < nr_parts; i++) {
        if(positions) 
            position = &(*positions)[i];
        if(velocities) 
            velocity = &(*velocities)[i];
        if(clusterIds) 
            clusterId = &(*clusterIds)[i];
        MxParticleHandle *p = MxParticle_New(types[i], position, velocity, clusterId);
        if(p) 
            result[i] = p->id;
    }

    return result;
}

std::vector<int> MxParticles_New(MxParticleType *type, 
                                 unsigned int nr_parts, 
                                 std::vector<MxVector3f> *positions, 
                                 std::vector<MxVector3f> *velocities, 
                                 std::vector<int> *clusterIds) 
{
    if(nr_parts == 0) {
        if(positions) 
            nr_parts = positions->size();
        else if(velocities) 
            nr_parts = velocities->size();
        else if(clusterIds) 
            nr_parts = clusterIds->size();
        else {
            mx_exp(std::runtime_error("Number of particles to create could not be determined."));
            return {};
        }
    }

    return MxParticles_New(std::vector<MxParticleType*>(nr_parts, type), positions, velocities, clusterIds);
}


HRESULT MxParticle_Become(MxParticle *part, MxParticleType *type) {
    HRESULT hr;
    if(!part || !type) {
        return mx_error(E_FAIL, "null arguments");
    }
    MxParticleHandle *pypart = part->py_particle();
    
    MxParticleType *currentType = &_Engine.types[part->typeId];
    
    assert(pypart->typeId == currentType->id);
    
    if(!SUCCEEDED(hr = currentType->del_part(part->id))) {
        return hr;
    };
    
    if(!SUCCEEDED(hr = type->addpart(part->id))) {
        return hr;
    }
    
    pypart->typeId = type->id;
    
    part->typeId = type->id;
    
    part->flags = type->particle_flags;
    
    if(!part->style) {
        bool visible = type->style->flags & STYLE_VISIBLE;
        if(visible != currentType->style->flags & STYLE_VISIBLE) {
            if(part->flags & PARTICLE_LARGE) _Engine.s.nr_visible_large_parts += visible ? 1 : -1;
            else _Engine.s.nr_visible_parts += visible ? 1 : -1;
        }
    }
    
    if(part->state_vector) {
        MxStateVector *oldState = part->state_vector;
        
        if(type->species) {
            part->state_vector = new MxStateVector(type->species, part, oldState);
        }
        else {
            part->state_vector = NULL;
        }
        
    }
    
    assert(type == &_Engine.types[part->typeId]);
    
    // TODO: bad things will happen if we convert between cluster and atomic types.
    
    return S_OK;
}

HRESULT MxParticleHandle::become(MxParticleType *type) {
    PARTICLE_SELFW(this, NULL)
    return MxParticle_Become(self, type);
}

MxParticleList *MxParticleHandle::neighbors(const float *distance, const std::vector<MxParticleType> *types) {
    try {
        PARTICLE_SELFW(this, NULL)
        
        float radius = distance ? *distance : _Engine.s.cutoff;

        std::set<short int> typeIds;
        if(types) for (auto &type : *types) typeIds.insert(type.id);
        else for(int i = 0; i < _Engine.nr_types; ++i) typeIds.insert(_Engine.types[i].id);
        
        // take into account the radius of this particle.
        radius += self->radius;
        
        uint16_t nr_parts = 0;
        int32_t *parts = NULL;
        
        MxParticle_Neighbors(self, radius, &typeIds, &nr_parts, &parts);
        
        MxParticleList *result = new MxParticleList(nr_parts, parts);
        if(parts) std::free(parts);
        return result;
    }
    catch(std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxParticleList *MxParticleHandle::getBondedNeighbors() {
    try {
        PARTICLE_SELFW(this, NULL)

        auto id = self->id;
        
        MxParticleList *list = new MxParticleList(5);
        
        for(int i = 0; i < _Engine.nr_bonds; ++i) {
            MxBond *b = &_Engine.bonds[i];
            if(b->flags & BOND_ACTIVE) {
                if(b->i == id) {
                    list->insert(b->j);
                }
                else if(b->j == id) {
                    list->insert(b->i);
                }
            }
        }
        return list;
    }
    catch(std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

std::vector<MxBondHandle*> *MxParticleHandle::getBonds() {
    try {
        PARTICLE_SELFW(this, 0)

        auto id = self->id;
        
        std::vector<MxBondHandle*> *bonds = new std::vector<MxBondHandle*>();
        
        for(int i = 0; i < _Engine.nr_bonds; ++i) {
            MxBond *b = &_Engine.bonds[i];
            if((b->flags & BOND_ACTIVE) && (b->i == id || b->j == id)) {
                bonds->push_back(new MxBondHandle(i));
            }
        }
        return bonds;
    }
    catch(std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

std::vector<MxAngleHandle*> *MxParticleHandle::getAngles() {
    try {
        PARTICLE_SELFW(this, 0)

        auto id = self->id;
        
        std::vector<MxAngleHandle*> *angles = new std::vector<MxAngleHandle*>();
        
        for(int i = 0; i < _Engine.nr_angles; ++i) {
            MxAngle *a = &_Engine.angles[i];
            if((a->flags & ANGLE_ACTIVE) && (a->i == id || a->j == id || a->k == id)) {
                angles->push_back(new MxAngleHandle(i));
            }
        }
        return angles;
    }
    catch(std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

std::vector<MxDihedralHandle*> *MxParticleHandle::getDihedrals() {
    try {
        PARTICLE_SELFW(this, 0)

        auto id = self->id;
        
        std::vector<MxDihedralHandle*> *dihedrals = new std::vector<MxDihedralHandle*>();
        
        for(int i = 0; i < _Engine.nr_dihedrals; ++i) {
            MxDihedral *d = &_Engine.dihedrals[i];
            if((d->i == id || d->j == id || d->k == id || d->l == id)) {
                dihedrals->push_back(new MxDihedralHandle(i));
            }
        }
        return dihedrals;
    }
    catch(std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

static MxParticleList *particletype_items(MxParticleType *self) {
    return &self->parts;
}

MxParticleList *MxParticleType::items() {
    return &parts;
}

float MxParticleHandle::distance(MxParticleHandle *_other) {
    PARTICLE_SELF(this)
    auto other = particleSelf(_other);
    
    if(other == NULL) {
        mx_error(E_FAIL, "invalid args, distance(Particle)");
        return NULL;
    }
    
    MxVector3f pos = self->global_position();
    MxVector3f opos = other->global_position();
    return (opos - pos).length();
}


int particle_init(MxParticleHandle *self, 
                  MxVector3f *position,
                  MxVector3f *velocity,
                  int *cluster) 
{
    
    try {
        Log(LOG_TRACE);

        PARTICLE_TYPE(self)
        
        MxRandomType &MxRandom = MxRandomEngine();

        MxVector3f _position;
        if (position) _position = *position;
        else {
            // make a random initial position
            auto eng_origin = engine_origin();
            auto eng_dims = engine_dimensions();
            std::uniform_real_distribution<float> x(eng_origin[0], eng_dims[0]);
            std::uniform_real_distribution<float> y(eng_origin[1], eng_dims[1]);
            std::uniform_real_distribution<float> z(eng_origin[2], eng_dims[2]);
            _position = {x(MxRandom), y(MxRandom), z(MxRandom)};
        }

        MxVector3f _velocity;
        if (velocity) _velocity = *velocity;
        else if(ptype->target_energy <= 0) _velocity = {0.0, 0.0, 0.0};
        else {
            // initial velocity, chosen to fit target temperature
            std::uniform_real_distribution<float> v(-1.0, 1.0);
            _velocity = {v(MxRandom), v(MxRandom), v(MxRandom)};
            float v2 = _velocity.dot();
            float x2 = (ptype->target_energy * 2. / (ptype->mass * v2));
            _velocity *= std::sqrt(x2);
        }
        
        // particle_init_ex will allocate a new particle, this can re-assign the pointers in
        // the engine particles, so need to pass cluster by id.
        int _clusterId = cluster ? *cluster : -1;
        
        return particle_init_ex(self, _position, _velocity, _clusterId);
        
    }
    catch (const std::exception &e) {
        return mx_exp(e);
    }
}

int particle_init_ex(MxParticleHandle *self,  const MxVector3f &position,
                     const MxVector3f &velocity,
                     int clusterId) {
    
    PARTICLE_TYPE(self)
    
    MxParticle part;
    bzero(&part, sizeof(MxParticle));
    part.radius = ptype->radius;
    part.mass = ptype->mass;
    part.imass = ptype->imass;
    part.id = engine_next_partid(&_Engine);
    part.typeId = ptype->id;
    part.flags = ptype->particle_flags;
    part.creation_time = _Engine.time;
    part.clusterId = clusterId;
    
    if(ptype->species) {
        part.state_vector = new MxStateVector(ptype->species, self);
    }
    
    if(ptype->isCluster()) {
        Log(LOG_DEBUG) << "making cluster";
    }
    
    part.position = position;
    part.velocity = velocity;
    
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }
    
    MxParticle *p = NULL;
    double pos[] = {part.position[0], part.position[1], part.position[2]};
    int result = engine_addpart (&_Engine, &part, pos, &p);
    
    if(result < 0) {
        std::string err = "error engine_addpart, ";
        err += engine_err_msg[-engine_err];
        return mx_error(result, err.c_str());
    }
    
    self->id = p->id;
    
    if(clusterId >= 0) {
        MxParticle *cluster = _Engine.s.partlist[clusterId];
        MxCluster_AddParticle((MxCluster*)cluster, p);
    } else {
        p->clusterId = -1;
    }
    
    p->_pyparticle = self;
    
    return 0;
}

MxParticleType* MxParticleType_FindFromName(const char* name) {
    for(int i = 0; i < _Engine.nr_types; ++i) {
        MxParticleType *type = &_Engine.types[i];
        if(std::strncmp(name, type->name, sizeof(MxParticleType::name)) == 0) {
            return type;
        }
    }
    return NULL;
}


HRESULT MxParticle_Verify() {

    bool result = true;

    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        space_cell *cell = &_Engine.s.cells[cid];
        for (int pid = 0 ; pid < cell->count ; pid++ ) {
            MxParticle *p  = &cell->parts[pid];
            result = p->verify() && result;
        }
    }

    for (int pid = 0 ; pid < _Engine.s.largeparts.count ; pid++ ) {
        MxParticle *p  = &_Engine.s.largeparts.parts[pid];
        result = p->verify() && result;
    }

    return result ? S_OK : E_FAIL;
}


namespace mx { namespace io {

#define MXPARTICLEIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXPARTICLEIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxParticle &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 

    MxIOElement *fe;

    MXPARTICLEIOTOEASY(fe, "force", dataElement.force);
    MXPARTICLEIOTOEASY(fe, "force_i", dataElement.force_i);
    MXPARTICLEIOTOEASY(fe, "number_density", dataElement.number_density);
    MXPARTICLEIOTOEASY(fe, "velocity", dataElement.velocity);
    MXPARTICLEIOTOEASY(fe, "position", MxParticleHandle(dataElement.id, dataElement.typeId).getPosition());
    MXPARTICLEIOTOEASY(fe, "creation_time", dataElement.creation_time);
    MXPARTICLEIOTOEASY(fe, "persistent_force", dataElement.persistent_force);
    MXPARTICLEIOTOEASY(fe, "radius", dataElement.radius);
    MXPARTICLEIOTOEASY(fe, "mass", dataElement.mass);
    MXPARTICLEIOTOEASY(fe, "q", dataElement.q);
    MXPARTICLEIOTOEASY(fe, "p0", dataElement.p0);
    MXPARTICLEIOTOEASY(fe, "v0", dataElement.v0);
    MXPARTICLEIOTOEASY(fe, "xk0", dataElement.xk[0]);
    MXPARTICLEIOTOEASY(fe, "xk1", dataElement.xk[1]);
    MXPARTICLEIOTOEASY(fe, "xk2", dataElement.xk[2]);
    MXPARTICLEIOTOEASY(fe, "xk3", dataElement.xk[3]);
    MXPARTICLEIOTOEASY(fe, "vk0", dataElement.vk[0]);
    MXPARTICLEIOTOEASY(fe, "vk1", dataElement.vk[1]);
    MXPARTICLEIOTOEASY(fe, "vk2", dataElement.vk[2]);
    MXPARTICLEIOTOEASY(fe, "vk3", dataElement.vk[3]);
    MXPARTICLEIOTOEASY(fe, "id", dataElement.id);
    MXPARTICLEIOTOEASY(fe, "vid", dataElement.vid);
    MXPARTICLEIOTOEASY(fe, "typeId", dataElement.typeId);
    MXPARTICLEIOTOEASY(fe, "clusterId", dataElement.clusterId);
    MXPARTICLEIOTOEASY(fe, "flags", dataElement.flags);
    
    if(dataElement.nr_parts > 0) {
        std::vector<int32_t> parts;
        for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
            parts.push_back(dataElement.parts[i]);
        MXPARTICLEIOTOEASY(fe, "parts", parts);
    }
    
    if(dataElement.style != NULL) {
        MXPARTICLEIOTOEASY(fe, "style", *dataElement.style);
    }
    if(dataElement.state_vector != NULL) {
        MXPARTICLEIOTOEASY(fe, "state_vector", *dataElement.state_vector);
    }

    fileElement->type = "Particle";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticle *dataElement) { 

    MxIOChildMap::const_iterator feItr;

    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "force", &dataElement->force);
    if(metaData.versionMajor > 0 || metaData.versionMinor > 31 || (metaData.versionMinor == 31 && metaData.versionPatch > 0)) 
        MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "force_i", &dataElement->force_i);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "velocity", &dataElement->velocity);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "number_density", &dataElement->number_density);
    dataElement->inv_number_density = 1.f / dataElement->number_density;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "position", &dataElement->position);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "creation_time", &dataElement->creation_time);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "persistent_force", &dataElement->persistent_force);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "radius", &dataElement->radius);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "mass", &dataElement->mass);
    dataElement->imass = 1.f / dataElement->mass;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "q", &dataElement->q);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "p0", &dataElement->p0);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "v0", &dataElement->v0);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "xk0", &dataElement->xk[0]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "xk1", &dataElement->xk[1]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "xk2", &dataElement->xk[2]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "xk3", &dataElement->xk[3]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "vk0", &dataElement->vk[0]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "vk1", &dataElement->vk[1]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "vk2", &dataElement->vk[2]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "vk3", &dataElement->vk[3]);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "vid", &dataElement->vid);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "typeId", &dataElement->typeId);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "clusterId", &dataElement->clusterId);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "flags", &dataElement->flags);
    
    // Skipping importing constituent particles; deduced from clusterId during import
    
    feItr = fileElement.children.find("style");
    if(feItr != fileElement.children.end()) {
        dataElement->style = new MxStyle();
        if(fromFile(*feItr->second, metaData, dataElement->style) != S_OK) 
            return E_FAIL;
    } 
    else dataElement->style = NULL;
    
    feItr = fileElement.children.find("state_vector");
    if(feItr != fileElement.children.end()) {
        dataElement->state_vector = NULL;
        if(fromFile(*feItr->second, metaData, &dataElement->state_vector) != S_OK) 
            return E_FAIL;
        dataElement->state_vector->owner = dataElement;
    }
    else dataElement->state_vector = NULL;

    return S_OK;
}

template <>
HRESULT toFile(const MxParticleType &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 

    MxIOElement *fe;

    MXPARTICLEIOTOEASY(fe, "id", dataElement.id);
    MXPARTICLEIOTOEASY(fe, "type_flags", dataElement.type_flags);
    MXPARTICLEIOTOEASY(fe, "particle_flags", dataElement.particle_flags);
    MXPARTICLEIOTOEASY(fe, "mass", dataElement.mass);
    MXPARTICLEIOTOEASY(fe, "charge", dataElement.charge);
    MXPARTICLEIOTOEASY(fe, "radius", dataElement.radius);
    MXPARTICLEIOTOEASY(fe, "kinetic_energy", dataElement.kinetic_energy);
    MXPARTICLEIOTOEASY(fe, "potential_energy", dataElement.potential_energy);
    MXPARTICLEIOTOEASY(fe, "target_energy", dataElement.target_energy);
    MXPARTICLEIOTOEASY(fe, "minimum_radius", dataElement.minimum_radius);
    MXPARTICLEIOTOEASY(fe, "eps", dataElement.eps);
    MXPARTICLEIOTOEASY(fe, "rmin", dataElement.rmin);
    MXPARTICLEIOTOEASY(fe, "dynamics", (unsigned int)dataElement.dynamics);
    MXPARTICLEIOTOEASY(fe, "name", std::string(dataElement.name));
    MXPARTICLEIOTOEASY(fe, "name2", std::string(dataElement.name2));
    if(dataElement.parts.nr_parts > 0) 
        MXPARTICLEIOTOEASY(fe, "parts", dataElement.parts);
    if(dataElement.types.nr_parts > 0) 
        MXPARTICLEIOTOEASY(fe, "types", dataElement.types);
    if(dataElement.style != NULL) {
        MXPARTICLEIOTOEASY(fe, "style", *dataElement.style);
    }
    if(dataElement.species != NULL) {
        MXPARTICLEIOTOEASY(fe, "species", *dataElement.species);
    }

    fileElement->type = "ParticleType";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticleType *dataElement) { 

    MxIOChildMap::const_iterator feItr;

    // Id set during registration: type ids are not preserved during import
    // MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "id", &dataElement->id);

    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "type_flags", &dataElement->type_flags);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "particle_flags", &dataElement->particle_flags);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "mass", &dataElement->mass);
    dataElement->imass = 1.f / dataElement->mass;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "charge", &dataElement->charge);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "radius", &dataElement->radius);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "kinetic_energy", &dataElement->kinetic_energy);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "potential_energy", &dataElement->potential_energy);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "target_energy", &dataElement->target_energy);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "minimum_radius", &dataElement->minimum_radius);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "eps", &dataElement->eps);
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "rmin", &dataElement->rmin);
    
    unsigned int dynamics;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "dynamics", &dynamics);
    dataElement->dynamics = dynamics;
    
    std::string name;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "name", &name);
    std::strncpy(dataElement->name, std::string(name).c_str(), MxParticleType::MAX_NAME);
    
    std::string name2;
    MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "name2", &name2);
    std::strncpy(dataElement->name2, std::string(name2).c_str(), MxParticleType::MAX_NAME);
    
    // Parts must be manually added, since part ids are not preserved during import
    // MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "parts", &dataElement->parts);
    
    if(fileElement.children.find("types") != fileElement.children.end()) 
        MXPARTICLEIOFROMEASY(feItr, fileElement.children, metaData, "types", &dataElement->types);
    
    feItr = fileElement.children.find("style");
    if(feItr != fileElement.children.end()) { 
        dataElement->style = new MxStyle();
        if(fromFile(*feItr->second, metaData, dataElement->style) != S_OK) 
            return E_FAIL;
    } 
    else {
        auto c = Magnum::Color3::fromSrgb(colors[(dataElement->id - 1) % (sizeof(colors)/sizeof(unsigned))]);
        dataElement->style = new MxStyle(&c);
    }
    
    feItr = fileElement.children.find("species");
    if(feItr != fileElement.children.end()) {
        dataElement->species = new MxSpeciesList();
        if(fromFile(*feItr->second, metaData, dataElement->species) != S_OK) 
            return E_FAIL;
    } 
    else dataElement->species = NULL;

    return S_OK;
}

}};

std::string MxParticle::toString() {
    return mx::io::toString(*this);
}

MxParticle *MxParticle::fromString(const std::string &str) {
    return new MxParticle(mx::io::fromString<MxParticle>(str));
}

std::string MxParticleType::toString() {
    return mx::io::toString(*this);
}

MxParticleType *MxParticleType::fromString(const std::string &str) {
    return new MxParticleType(mx::io::fromString<MxParticleType>(str));
}
