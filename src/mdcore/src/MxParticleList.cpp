/*
 * MxParticleList.cpp
 *
 *  Created on: Nov 23, 2020
 *      Author: andy
 */


#include <MxParticleList.hpp>
#include "engine.h"
#include <metrics.h>
#include "../../mx_error.h"
#include "../../io/MxFIO.h"
#include <cstdarg>
#include <iostream>


void MxParticleList::free()
{
}

uint16_t MxParticleList::insert(int32_t id)
{
    /* do we need to extend the partlist? */
    if ( nr_parts == size_parts ) {
        size_parts += space_partlist_incr;
        int32_t* temp = NULL;
        if (( temp = (int32_t*)malloc( sizeof(int32_t) * size_parts )) == NULL ) {
            return mx_error(E_FAIL, "could not allocate space for type particles");
        }
        memcpy( temp , parts , sizeof(int32_t) * nr_parts );
        ::free( parts );
        parts = temp;
    }
    
    parts[nr_parts] = id;

    return nr_parts++;
}

uint16_t MxParticleList::insert(const MxParticleHandle *particle) {
    if(particle) return insert(particle->id);

    mx_error(E_FAIL, "cannot insert a NULL particle");
    return 0;
}

uint16_t MxParticleList::remove(int32_t id)
{
    int i = 0;
    for(; i < nr_parts; i++) {
        if(parts[i] == id)
            break;
    }
    
    if(i == nr_parts) {
        return mx_error(E_FAIL, "type does not contain particle id");
    }
    
    nr_parts--;
    if(i < nr_parts) {
        parts[i] = parts[nr_parts];
    }
    
    return i;
}

void MxParticleList::extend(const MxParticleList &other) {
    for(int i = 0; i < other.nr_parts; ++i) this->insert(other.parts[i]);
}

MxParticleHandle *MxParticleList::item(const int32_t &i) {
    if(i < nr_parts) {
        MxParticle *part = _Engine.s.partlist[parts[i]];
        if(part) {
            return part->py_particle();
        }
    }
    else {
        throw std::runtime_error("index out of range");
    }
    return NULL;
}

MxParticleList::MxParticleList() : 
    flags(PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF), 
    size_parts(0), 
    nr_parts(0)
{
    this->parts = (int32_t*)malloc(this->size_parts * sizeof(int32_t));
}

MxParticleList::MxParticleList(uint16_t init_size, uint16_t flags) : MxParticleList() {
    this->flags = flags;
    this->size_parts = init_size;
    ::free(this->parts);
    this->parts = (int32_t*)malloc(init_size * sizeof(int32_t));
}

MxParticleList::MxParticleList(MxParticleHandle *part) : 
    MxParticleList(1, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    if(!part) throw std::runtime_error("Cannot instance a list from NULL handle");
    
    MxParticle *p = part->part();
    if(!p) throw std::runtime_error("Cannot instance a list from NULL particle");

    this->nr_parts = 1;
    this->parts[0] = p->id;
}

MxParticleList::MxParticleList(std::vector<MxParticleHandle*> particles) : 
    MxParticleList(particles.size(), PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = particles.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = particles[i]->part();
        if(!p) {
            throw std::runtime_error("Cannot initialize a list with a NULL particle");
        }
        this->parts[i] = p->id;
    }
}

MxParticleList::MxParticleList(uint16_t nr_parts, int32_t *parts) : 
    MxParticleList(nr_parts, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = nr_parts;
    memcpy(this->parts, parts, nr_parts * sizeof(int32_t));
}

MxParticleList::MxParticleList(const MxParticleList &other) : 
    MxParticleList(other.nr_parts, other.parts)
{}

MxParticleList::~MxParticleList() {
    if(this->flags & PARTICLELIST_OWNDATA && size_parts > 0) {
        ::free(this->parts);
    }
}

// TODO: in universe.bind, check keywords are correct, and no extra keyworkds
// TODO: simulator init, turn off periodoc if only single cell.
MxMatrix3f MxParticleList::getVirial()
{
    try {
        MxMatrix3f result;
        if(SUCCEEDED(MxParticles_Virial(this->parts, this->nr_parts, 0, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }

    return MxMatrix3f();
}

float MxParticleList::getRadiusOfGyration() {
    try {
        float result;
        if(SUCCEEDED(MxParticles_RadiusOfGyration(this->parts, this->nr_parts, &result))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }
    return 0.0;
}

MxVector3f MxParticleList::getCenterOfMass() {
    try {
        MxVector3f result;
        if(SUCCEEDED(MxParticles_CenterOfMass(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }

    return MxVector3f();
}

MxVector3f MxParticleList::getCentroid() {
    try {
        MxVector3f result;
        if(SUCCEEDED(MxParticles_CenterOfGeometry(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }

    return MxVector3f();
}

MxMatrix3f MxParticleList::getMomentOfInertia() {
    try {
        MxMatrix3f result;
        if(SUCCEEDED(MxParticles_MomentOfInertia(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }

    return MxMatrix3f();
}

std::vector<MxVector3f> MxParticleList::getPositions() {
    std::vector<MxVector3f> result(this->nr_parts);
    
    try {
        for(int i = 0; i < this->nr_parts; ++i) {
            MxParticle *part = _Engine.s.partlist[this->parts[i]];
            MxVector3f pos = part->global_position();
            result[i] = MxVector3f(pos.x(), pos.y(), pos.z());
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<MxVector3f> MxParticleList::getVelocities() {
    std::vector<MxVector3f> result(this->nr_parts);
    
    try {
        for(int i = 0; i < this->nr_parts; ++i) {
            MxParticle *part = _Engine.s.partlist[this->parts[i]];
            result[i] = MxVector3f(part->velocity[0], part->velocity[1], part->velocity[2]);
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<MxVector3f> MxParticleList::getForces() {
    std::vector<MxVector3f> result(this->nr_parts);
    
    try{
        for(int i = 0; i < this->nr_parts; ++i) {
            MxParticle *part = _Engine.s.partlist[this->parts[i]];
            result[i] = MxVector3f(part->force[0], part->force[1], part->force[2]);
        }
    }
    catch(const std::exception &e) {
        mx_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<MxVector3f> MxParticleList::sphericalPositions(MxVector3f *origin) {
    std::vector<MxVector3f> result(this->nr_parts);

    MxVector3f _origin;
    if(origin) _origin = *origin;
    else{
        auto center = MxVector3f((float)_Engine.s.dim[0], (float)_Engine.s.dim[1], (float)_Engine.s.dim[2]);
        _origin = center / 2;
    }
    
    for(int i = 0; i < this->nr_parts; ++i) {
        MxParticle *part = _Engine.s.partlist[this->parts[i]];
        MxVector3f pos = part->global_position();
        result[i] = MxCartesianToSpherical(pos, *origin);
    }
    
    return result;
}

MxParticleList *MxParticleList::pack(size_t n, ...)
{
    int i;
    MxParticleList *result = new MxParticleList(n, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF);
    va_list vargs;
    
    va_start(vargs, n);
    if (result == NULL) {
        va_end(vargs);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        int o = va_arg(vargs, int);
        result->parts[i] = o;
    }
    va_end(vargs);
    return result;
}

MxParticleList* MxParticleList::all() {
    MxParticleList* list = new MxParticleList(_Engine.s.nr_parts);
    
    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        space_cell *cell = &_Engine.s.cells[cid];
        for (int pid = 0 ; pid < cell->count ; pid++ ) {
            MxParticle *p  = &cell->parts[pid];
            list->insert(p->id);
        }
    }
    
    for (int pid = 0 ; pid < _Engine.s.largeparts.count ; pid++ ) {
        MxParticle *p  = &_Engine.s.largeparts.parts[pid];
        list->insert(p->id);
    }
    
    return list;
}

std::string MxParticleList::toString() {
    return mx::io::toString(*this);
}

MxParticleList *MxParticleList::fromString(const std::string &str) {
    return new MxParticleList(mx::io::fromString<MxParticleList>(str));
}


namespace mx { namespace io {

template <>
HRESULT toFile(const MxParticleList &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 

    MxIOElement *fe;

    std::vector<int32_t> parts;
    for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
        parts.push_back(dataElement.parts[i]);
    fe = new MxIOElement();
    if(toFile(parts, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["parts"] = fe;

    fileElement->type = "ParticleList";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticleList *dataElement) { 

    MxIOChildMap::const_iterator feItr;
    std::vector<int32_t> parts;

    feItr = fileElement.children.find("parts");
    if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &parts) != S_OK) 
        return E_FAIL;

    for(unsigned int i = 0; i < parts.size(); i++) 
        dataElement->insert(parts[i]);

    dataElement->flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF;

    return S_OK;
}

}};
