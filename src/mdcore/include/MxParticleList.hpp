/*
 * MxParticleList.h
 *
 *  Created on: Nov 23, 2020
 *      Author: andy
 */

#ifndef _MDCORE_MXPARTICLELIST_H_
#define _MDCORE_MXPARTICLELIST_H_

#include "mx_port.h"
#include "../../types/mx_types.h"

#include <vector>

enum ParticleListFlags {
    // list owns the data the MxParticleList::parts
    PARTICLELIST_OWNDATA = 1 << 0,
    
    // list supports insertion / deletion
    PARTICLELIST_MUTABLE = 1 << 1,
    
    // list owns it's own data, it was allocated.
    PARTICLELIST_OWNSELF = 1 << 2,
};

struct MxParticleHandle;

/** The #potential structure. */
struct CAPI_EXPORT MxParticleList {
    int32_t *parts;
    int32_t nr_parts;
    int32_t size_parts;
    uint16_t flags;
    
    // frees the memory associated with the parts list.
    void free();
    
    // inserts the given id into the list, returns the index of the item. 
    uint16_t insert(int32_t item);
    uint16_t insert(const MxParticleHandle *particle);
    
    // looks for the item with the given id and deletes it form the list
    uint16_t remove(int32_t id);
    
    // inserts the contents of another list
    void extend(const MxParticleList &other);

    // looks for the item at the given index and returns it if found, otherwise returns NULL
    MxParticleHandle *item(const int32_t &i);

    // packs a variable number of particle ids into a new list
    static MxParticleList *pack(size_t n, ...);

    // returns an instance populated with all current particles
    static MxParticleList* all();

    MxMatrix3f getVirial();
    float getRadiusOfGyration();
    MxVector3f getCenterOfMass();
    MxVector3f getCentroid();
    MxMatrix3f getMomentOfInertia();
    std::vector<MxVector3f> getPositions();
    std::vector<MxVector3f> getVelocities();
    std::vector<MxVector3f> getForces();

    std::vector<MxVector3f> sphericalPositions(MxVector3f *origin=NULL);

    MxParticleList();
    MxParticleList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
    MxParticleList(MxParticleHandle *part);
    MxParticleList(std::vector<MxParticleHandle*> particles);
    MxParticleList(uint16_t nr_parts, int32_t *parts);
    MxParticleList(const MxParticleList &other);
    ~MxParticleList();
    
};

struct MxParticleType;
struct MxClusterParticleType;

struct CAPI_EXPORT MxParticleTypeList {
    int32_t *parts;
    int32_t nr_parts;
    int32_t size_parts;
    uint16_t flags;
    
    // frees the memory associated with the parts list.
    void free();
    
    // inserts the given id into the list, returns the index of the item. 
    uint16_t insert(int32_t item);
    uint16_t insert(const MxParticleType *ptype);
    
    // looks for the item with the given id and deletes it form the list
    uint16_t remove(int32_t id);
    
    // inserts the contents of another list
    void extend(const MxParticleTypeList &other);

    // looks for the item at the given index and returns it if found, otherwise returns NULL
    MxParticleType *item(const int32_t &i);

    // packs a variable number of particle type ids into a new list
    static MxParticleTypeList *pack(size_t n, ...);

    // returns an instance populated with particles of all current particle types
    MxParticleList *particles();

    // returns an instance populated with all current particle types
    static MxParticleTypeList *all();

    MxMatrix3f getVirial();
    float getRadiusOfGyration();
    MxVector3f getCenterOfMass();
    MxVector3f getCentroid();
    MxMatrix3f getMomentOfInertia();
    std::vector<MxVector3f> getPositions();
    std::vector<MxVector3f> getVelocities();
    std::vector<MxVector3f> getForces();

    std::vector<MxVector3f> sphericalPositions(MxVector3f *origin=NULL);

    MxParticleTypeList();
    MxParticleTypeList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
    MxParticleTypeList(MxParticleType *ptype);
    MxParticleTypeList(std::vector<MxParticleType*> ptypes);
    MxParticleTypeList(uint16_t nr_parts, int32_t *ptypes);
    MxParticleTypeList(const MxParticleTypeList &other);
    ~MxParticleTypeList();
    
};

#endif /* SRC_MDCORE_SRC_MXPARTICLELIST_H_ */
