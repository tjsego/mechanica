/**
 * @file MxParticleTypeList.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines the particle tye list
 * @date 2021-08-19
 * 
 */
#ifndef _MDCORE_MXPARTICLETYPELIST_H_
#define _MDCORE_MXPARTICLETYPELIST_H_

#include "MxParticleList.hpp"

struct MxParticleType;
struct MxClusterParticleType;

/**
 * @brief A special list with convenience methods 
 * for working with sets of particle types.
 */
struct CAPI_EXPORT MxParticleTypeList {
    int32_t *parts;
    int32_t nr_parts;
    int32_t size_parts;
    uint16_t flags;
    
    // frees the memory associated with the parts list.
    void free();

    /**
     * @brief Reserve enough storage for a given number of items.
     * 
     * @param _nr_parts number of items
     * @return HRESULT 
     */
    HRESULT reserve(size_t _nr_parts);
    
    // inserts the given id into the list, returns the index of the item. 
    uint16_t insert(int32_t item);

    /**
     * @brief Inserts the given particle type into the list, returns the index of the item. 
     * 
     * @param ptype 
     * @return uint16_t 
     */
    uint16_t insert(const MxParticleType *ptype);
    
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
    void extend(const MxParticleTypeList &other);

    /**
     * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
     * 
     * @param i index of lookup
     * @return MxParticleType* 
     */
    MxParticleType *item(const int32_t &i);

    // packs a variable number of particle type ids into a new list
    static MxParticleTypeList *pack(size_t n, ...);

    /**
     * @brief returns a list populated with particles of all current particle types
     * 
     * @return MxParticleList* 
     */
    MxParticleList *particles();

    /**
     * @brief returns an instance populated with all current particle types
     * 
     * @return MxParticleTypeList* 
     */
    static MxParticleTypeList *all();

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

    MxParticleTypeList();
    MxParticleTypeList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
    MxParticleTypeList(MxParticleType *ptype);
    MxParticleTypeList(std::vector<MxParticleType*> ptypes);
    MxParticleTypeList(uint16_t nr_parts, int32_t *ptypes);
    MxParticleTypeList(const MxParticleTypeList &other);
    ~MxParticleTypeList();

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
     * @return MxParticleTypeList* 
     */
    static MxParticleTypeList *fromString(const std::string &str);
    
};

namespace mx { namespace io {

template <>
HRESULT toFile(const MxParticleTypeList &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxParticleTypeList *dataElement);

}};

#endif //_MDCORE_MXPARTICLETYPELIST_H_