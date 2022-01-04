/**
 * @file MxParticleTypeList.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines the particle tye list
 * @date 2021-08-19
 * 
 */
#include "MxParticleTypeList.h"

#include "engine.h"

void MxParticleTypeList::free() {}

uint16_t MxParticleTypeList::insert(int32_t item) {
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
    
    parts[nr_parts] = item;

    return nr_parts++;
}

uint16_t MxParticleTypeList::insert(const MxParticleType *ptype) {
    if(ptype) return insert(ptype->id);

    mx_error(E_FAIL, "cannot insert a NULL type");
}

uint16_t MxParticleTypeList::remove(int32_t id) {
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

void MxParticleTypeList::extend(const MxParticleTypeList &other) {
    for(int i = 0; i < other.nr_parts; ++i) this->insert(other.parts[i]);
}

MxParticleType *MxParticleTypeList::item(const int32_t &i) {
    if(i < nr_parts) {
        return &_Engine.types[parts[i]];
    }
    else {
        throw std::runtime_error("index out of range");
    }
    return NULL;
}

MxParticleTypeList *MxParticleTypeList::pack(size_t n, ...) {
    int i;
    MxParticleTypeList *result = new MxParticleTypeList(n, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF);
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

MxParticleList *MxParticleTypeList::particles() {
    MxParticleList *list = new MxParticleList();

    for(int tid = 0; tid < this->nr_parts; ++tid) list->extend(this->item(tid)->parts);

    return list;
}

MxParticleTypeList* MxParticleTypeList::all() {
    MxParticleTypeList* list = new MxParticleTypeList();
    
    for(int tid = 0; tid < _Engine.nr_types; tid++) list->insert(tid);
    
    return list;
}

MxMatrix3f MxParticleTypeList::getVirial() {
    auto p = this->particles();
    auto r = p->getVirial();
    delete p;
    return r;
}

float MxParticleTypeList::getRadiusOfGyration() {
    auto p = this->particles();
    auto r = p->getRadiusOfGyration();
    delete p;
    return r;
}

MxVector3f MxParticleTypeList::getCenterOfMass() {
    auto p = this->particles();
    auto r = p->getCenterOfMass();
    delete p;
    return r;
}

MxVector3f MxParticleTypeList::getCentroid() {
    auto p = this->particles();
    auto r = p->getCentroid();
    delete p;
    return r;
}

MxMatrix3f MxParticleTypeList::getMomentOfInertia() {
    auto p = this->particles();
    auto r = p->getMomentOfInertia();
    delete p;
    return r;
}

std::vector<MxVector3f> MxParticleTypeList::getPositions() {
    auto p = this->particles();
    auto r = p->getPositions();
    delete p;
    return r;
}

std::vector<MxVector3f> MxParticleTypeList::getVelocities() {
    auto p = this->particles();
    auto r = p->getVelocities();
    delete p;
    return r;
}

std::vector<MxVector3f> MxParticleTypeList::getForces() {
    auto p = this->particles();
    auto r = p->getForces();
    delete p;
    return r;
}

std::vector<MxVector3f> MxParticleTypeList::sphericalPositions(MxVector3f *origin) {
    auto p = this->particles();
    auto r = p->sphericalPositions();
    delete p;
    return r;
}

MxParticleTypeList::MxParticleTypeList() : 
    flags(PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF), 
    size_parts(0), 
    nr_parts(0)
{
    this->parts = (int32_t*)malloc(this->size_parts * sizeof(int32_t));
}

MxParticleTypeList::MxParticleTypeList(uint16_t init_size, uint16_t flags) : MxParticleTypeList() {
    this->flags = flags;
    this->size_parts = init_size;
    ::free(this->parts);
    this->parts = (int32_t*)malloc(init_size * sizeof(int32_t));
}

MxParticleTypeList::MxParticleTypeList(MxParticleType *ptype) : 
    MxParticleTypeList(1, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    if(!ptype) throw std::runtime_error("Cannot instance a list from NULL type");
    
    this->nr_parts = 1;
    this->parts[0] = ptype->id;
}

MxParticleTypeList::MxParticleTypeList(std::vector<MxParticleType*> ptypes) : 
    MxParticleTypeList(ptypes.size(), PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = ptypes.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        MxParticleType *t = ptypes[i];
        if(!t) {
            throw std::runtime_error("Cannot initialize a list with a NULL type");
        }
        this->parts[i] = t->id;
    }
}

MxParticleTypeList::MxParticleTypeList(uint16_t nr_parts, int32_t *ptypes) : 
    MxParticleTypeList(nr_parts, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = nr_parts;
    memcpy(this->parts, parts, nr_parts * sizeof(int32_t));
}

MxParticleTypeList::MxParticleTypeList(const MxParticleTypeList &other) : 
    MxParticleTypeList(other.nr_parts, other.parts)
{}

MxParticleTypeList::~MxParticleTypeList() {
    if(this->flags & PARTICLELIST_OWNDATA && size_parts > 0) {
        ::free(this->parts);
    }
}


namespace mx { namespace io {

template <>
HRESULT toFile(const MxParticleTypeList &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) { 

    MxIOElement *fe;

    std::vector<int32_t> parts;
    for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
        parts.push_back(dataElement.parts[i]);
    fe = new MxIOElement();
    if(toFile(parts, metaData, fe) != S_OK) 
        return E_FAIL;
    fe->parent = fileElement;
    fileElement->children["parts"] = fe;

    fileElement->type = "ParticleTypeList";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticleTypeList *dataElement) { 

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
