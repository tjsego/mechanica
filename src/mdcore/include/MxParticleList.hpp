/*
 * MxParticleList.h
 *
 *  Created on: Nov 23, 2020
 *      Author: andy
 */

#ifndef _MDCORE_MXPARTICLELIST_H_
#define _MDCORE_MXPARTICLELIST_H_

#include "mx_port.h"
#include "../../io/mx_io.h"
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

/**
 * @brief A special list with convenience methods 
 * for working with sets of particles.
 */
struct CAPI_EXPORT MxParticleList {
    int32_t *parts;
    int32_t nr_parts;
    int32_t size_parts;
    uint16_t flags;
    
    // frees the memory associated with the parts list.
    void free();
    
    // inserts the given id into the list, returns the index of the item. 
    uint16_t insert(int32_t item);

    /**
     * @brief Inserts the given particle into the list, returns the index of the item. 
     * 
     * @param particle particle to insert
     * @return uint16_t 
     */
    uint16_t insert(const MxParticleHandle *particle);
    
    /**
     * @brief looks for the item with the given id and deletes it form the list
     * 
     * @param id id to remove
     * @return uint16_t 
     */
    uint16_t remove(int32_t id);
    
    /**
     * @brief inserts the contents of another list
     * 
     * @param other another list
     */
    void extend(const MxParticleList &other);

    /**
     * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
     * 
     * @param i index of lookup
     * @return MxParticleHandle* 
     */
    MxParticleHandle *item(const int32_t &i);

    // packs a variable number of particle ids into a new list
    static MxParticleList *pack(size_t n, ...);

    /**
     * @brief returns an instance populated with all current particles
     * 
     * @return MxParticleList* 
     */
    static MxParticleList* all();

    MxMatrix3f getVirial();
    float getRadiusOfGyration();
    MxVector3f getCenterOfMass();
    MxVector3f getCentroid();
    MxMatrix3f getMomentOfInertia();
    std::vector<MxVector3f> getPositions();
    std::vector<MxVector3f> getVelocities();
    std::vector<MxVector3f> getForces();

    /**
     * @brief Get the spherical coordinates of each particle
     * 
     * @param origin optional origin of coordinates; default is center of universe
     * @return std::vector<MxVector3f> 
     */
    std::vector<MxVector3f> sphericalPositions(MxVector3f *origin=NULL);

    MxParticleList();
    MxParticleList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
    MxParticleList(MxParticleHandle *part);
    MxParticleList(std::vector<MxParticleHandle*> particles);
    MxParticleList(uint16_t nr_parts, int32_t *parts);
    MxParticleList(const MxParticleList &other);
    ~MxParticleList();

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation
     * 
     * @param str 
     * @return MxParticleList* 
     */
    static MxParticleList *fromString(const std::string &str);
    
};

namespace mx { namespace io {

template <>
HRESULT toFile(const MxParticleList &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticleList *dataElement);

}};

#endif /* SRC_MDCORE_SRC_MXPARTICLELIST_H_ */
