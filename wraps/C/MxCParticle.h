/**
 * @file MxCParticle.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxParticle and associated features
 * @date 2022-03-28
 */

#ifndef _WRAPS_C_MXCPARTICLE_H_
#define _WRAPS_C_MXCPARTICLE_H_

#include <mx_port.h>

#include "MxCStyle.h"
#include "MxCStateVector.h"
#include "MxCSpecies.h"

/**
 * @brief Particle type style definition in Mechanica C
 * 
 */
struct CAPI_EXPORT MxCParticleTypeStyle {
    char *color;
    unsigned int visible;
    char *speciesName;
    char *speciesMapName;
    float speciesMapMin;
    float speciesMapMax;
};

/**
 * @brief Particle type definition in Mechanica C
 * 
 */
struct CAPI_EXPORT MxCParticleType {
    double mass;
    double charge;
    double radius;
    double *target_energy;
    double minimum_radius;
    double eps;
    double rmin;
    unsigned char dynamics;
    unsigned int frozen;
    char *name;
    char *name2;
    struct MxCParticleTypeStyle *style;
    unsigned int numSpecies;
    char **species;
};

// Handles

struct CAPI_EXPORT MxParticleDynamicsEnumHandle {
    unsigned char PARTICLE_NEWTONIAN;
    unsigned char PARTICLE_OVERDAMPED;
};

struct CAPI_EXPORT MxParticleFlagsHandle {
    int PARTICLE_NONE;
    int PARTICLE_GHOST;
    int PARTICLE_CLUSTER;
    int PARTICLE_BOUND;
    int PARTICLE_FROZEN_X;
    int PARTICLE_FROZEN_Y;
    int PARTICLE_FROZEN_Z;
    int PARTICLE_FROZEN;
    int PARTICLE_LARGE;
};

/**
 * @brief Handle to a @ref MxParticleHandle instance
 * 
 */
struct CAPI_EXPORT MxParticleHandleHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxParticleType instance
 * 
 */
struct CAPI_EXPORT MxParticleTypeHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxParticleList instance
 * 
 */
struct CAPI_EXPORT MxParticleListHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxParticleTypeList instance
 * 
 */
struct CAPI_EXPORT MxParticleTypeListHandle {
    void *MxObj;
};


/////////////////////
// MxCParticleType //
/////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct MxCParticleType) MxCParticleTypeDef_init();


//////////////////////////
// MxCParticleTypeStyle //
//////////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct MxCParticleTypeStyle) MxCParticleTypeStyleDef_init();


////////////////////////
// MxParticleDynamics //
////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleDynamics_init(struct MxParticleDynamicsEnumHandle *handle);


/////////////////////
// MxParticleFlags //
/////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleFlags_init(struct MxParticleFlagsHandle *handle);


//////////////////////
// MxParticleHandle //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param pid particle id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_init(struct MxParticleHandleHandle *handle, unsigned int pid);

/**
 * @brief Destroys the handle instance
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_destroy(struct MxParticleHandleHandle *handle);

/**
 * @brief Gets the particle type of this handle. 
 * 
 * @param handle populated handle
 * @param typeHandle type handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getType(struct MxParticleHandleHandle *handle, struct MxParticleTypeHandle *typeHandle);

/**
 * @brief Splits a single particle into two. Returns the new particle. 
 * 
 * @param handle populated handle
 * @param newParticleHandle new particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_split(struct MxParticleHandleHandle *handle, struct MxParticleHandleHandle *newParticleHandle);

/**
 * @brief Destroys the particle, and removes it from inventory. 
 * 
 * Subsequent references to a destroyed particle result in an error.
 * @param handle 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_destroyParticle(struct MxParticleHandleHandle *handle);

/**
 * @brief Calculates the particle's coordinates in spherical coordinates relative to the center of the universe. 
 * 
 * @param handle populated handle
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_sphericalPosition(struct MxParticleHandleHandle *handle, float **position);

/**
 * @brief Calculates the particle's coordinates in spherical coordinates relative to a point
 * 
 * @param handle populated handle
 * @param origin relative point
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_sphericalPositionPoint(struct MxParticleHandleHandle *handle, float *origin, float **position);

/**
 * @brief Computes the relative position with respect to an origin while. 
 * 
 * @param handle populated handle
 * @param origin relative point
 * @param position 3-element allocated array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_relativePosition(struct MxParticleHandleHandle *handle, float *origin, float **position);

/**
 * @brief Dynamically changes the *type* of an object. We can change the type of a 
 * MxParticleType-derived object to anyther pre-existing MxParticleType-derived 
 * type. What this means is that if we have an object of say type 
 * *A*, we can change it to another type, say *B*, and and all of the forces 
 * and processes that acted on objects of type A stip and the forces and 
 * processes defined for type B now take over. 
 * 
 * @param handle populated handle
 * @param typeHandle new particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_become(struct MxParticleHandleHandle *handle, struct MxParticleTypeHandle *typeHandle);

/**
 * @brief Get the particles within a distance of a particle
 * 
 * @param handle populated handle
 * @param distance distance
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_neighborsD(struct MxParticleHandleHandle *handle, 
                                                float distance, 
                                                struct MxParticleHandleHandle **neighbors, 
                                                int *numNeighbors);

/**
 * @brief Get the particles of a set of types within the global cutoff distance
 * 
 * @param handle populated handle
 * @param ptypes particle types
 * @param numTypes number of particle types
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_neighborsT(struct MxParticleHandleHandle *handle, 
                                                struct MxParticleTypeHandle *ptypes, 
                                                int numTypes, 
                                                struct MxParticleHandleHandle **neighbors, 
                                                int *numNeighbors);

/**
 * @brief Get the particles of a set of types within a distance of a particle
 * 
 * @param handle populated handle
 * @param distance distance
 * @param ptypes particle types
 * @param numTypes number of particle types
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_neighborsDT(struct MxParticleHandleHandle *handle, 
                                                 float distance, 
                                                 struct MxParticleTypeHandle *ptypes, 
                                                 int numTypes, 
                                                 struct MxParticleHandleHandle **neighbors, 
                                                 int *numNeighbors);

/**
 * @brief Get a list of all bonded neighbors. 
 * 
 * @param handle populated handle
 * @param neighbors neighbors
 * @param numNeighbors number of neighbors
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getBondedNeighbors(struct MxParticleHandleHandle *handle, 
                                                        struct MxParticleHandleHandle **neighbors, 
                                                        int *numNeighbors);

/**
 * @brief Calculates the distance to another particle
 * 
 * @param handle populated handle
 * @param other populated handle of another particle
 * @param distance distance
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_distance(struct MxParticleHandleHandle *handle, struct MxParticleHandleHandle *other, float *distance);

/**
 * @brief Get the particle mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getMass(struct MxParticleHandleHandle *handle, double *mass);

/**
 * @brief Set the particle mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setMass(struct MxParticleHandleHandle *handle, double mass);

/**
 * @brief Test whether the particle is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getFrozen(struct MxParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setFrozen(struct MxParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getFrozenX(struct MxParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setFrozenX(struct MxParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getFrozenY(struct MxParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setFrozenY(struct MxParticleHandleHandle *handle, bool frozen);

/**
 * @brief Test whether the particle is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getFrozenZ(struct MxParticleHandleHandle *handle, bool *frozen);

/**
 * @brief Set whether the particle is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setFrozenZ(struct MxParticleHandleHandle *handle, bool frozen);

/**
 * @brief Get the particle style. Fails if no style is set.
 * 
 * @param handle populated handle
 * @param style handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getStyle(struct MxParticleHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Test whether the particle has a style
 * 
 * @param handle populated handle
 * @param hasStyle flag signifying whether the particle has a style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_hasStyle(struct MxParticleHandleHandle *handle, bool *hasStyle);

/**
 * @brief Set the particle style
 * 
 * @param handle populated handle
 * @param style style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setStyle(struct MxParticleHandleHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Get the particle age
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getAge(struct MxParticleHandleHandle *handle, double *age);

/**
 * @brief Get the particle radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getRadius(struct MxParticleHandleHandle *handle, double *radius);

/**
 * @brief Set the particle radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setRadius(struct MxParticleHandleHandle *handle, double radius);

/**
 * @brief Get the particle name
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getName(struct MxParticleHandleHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Get the particle second name
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getName2(struct MxParticleHandleHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Get the particle position
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getPosition(struct MxParticleHandleHandle *handle, float **position);

/**
 * @brief Set the particle position
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setPosition(struct MxParticleHandleHandle *handle, float *position);

/**
 * @brief Get the particle velocity
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getVelocity(struct MxParticleHandleHandle *handle, float **velocity);

/**
 * @brief Set the particle velocity
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setVelocity(struct MxParticleHandleHandle *handle, float *velocity);

/**
 * @brief Get the particle force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getForce(struct MxParticleHandleHandle *handle, float **force);

/**
 * @brief Get the particle initial force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getForceInit(struct MxParticleHandleHandle *handle, float **force);

/**
 * @brief Set the particle initial force
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setForceInit(struct MxParticleHandleHandle *handle, float *force);

/**
 * @brief Get the particle id
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getId(struct MxParticleHandleHandle *handle, int *pid);

/**
 * @brief Get the particle type id
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getTypeId(struct MxParticleHandleHandle *handle, int *tid);

/**
 * @brief Get the particle cluster id. -1 if particle is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getClusterId(struct MxParticleHandleHandle *handle, int *cid);

/**
 * @brief Get the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getFlags(struct MxParticleHandleHandle *handle, int *flags);

/**
 * @brief Test whether a particle has species
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_hasSpecies(struct MxParticleHandleHandle *handle, bool *flag);

/**
 * @brief Get the state vector. Fails if the particle does not have a state vector
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_getSpecies(struct MxParticleHandleHandle *handle, struct MxStateVectorHandle *svec);

/**
 * @brief Set the state vector. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_setSpecies(struct MxParticleHandleHandle *handle, struct MxStateVectorHandle *svec);

/**
 * @brief Convert the particle to a cluster; fails if particle is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_toCluster(struct MxParticleHandleHandle *handle, struct MxClusterParticleHandleHandle *chandle);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleHandle_toString(struct MxParticleHandleHandle *handle, char **str, unsigned int *numChars);


////////////////////
// MxParticleType //
////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_init(struct MxParticleTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param pdef definition
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleType_initD(struct MxParticleTypeHandle *handle, struct MxCParticleType pdef);

/**
 * @brief Get the type name.
 * 
 * @param handle populated handle
 * @param name type name
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getName(struct MxParticleTypeHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Set the type name. Throws an error if the type is already registered.
 * 
 * @param handle populated handle
 * @param name type name
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setName(struct MxParticleTypeHandle *handle, const char *name);

/**
 * @brief Get the type id. -1 if not registered.
 * 
 * @param handle populated handle
 * @param id type id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getId(struct MxParticleTypeHandle *handle, int *id);

/**
 * @brief Get the type flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getTypeFlags(struct MxParticleTypeHandle *handle, unsigned int *flags);

/**
 * @brief Set the type flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setTypeFlags(struct MxParticleTypeHandle *handle, unsigned int flags);

/**
 * @brief Get the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getParticleFlags(struct MxParticleTypeHandle *handle, unsigned int *flags);

/**
 * @brief Set the particle flags
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setParticleFlags(struct MxParticleTypeHandle *handle, unsigned int flags);

/**
 * @brief Get the type style. 
 * 
 * @param handle populated handle
 * @param style handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getStyle(struct MxParticleTypeHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Set the type style
 * 
 * @param handle populated handle
 * @param style style
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setStyle(struct MxParticleTypeHandle *handle, struct MxStyleHandle *style);

/**
 * @brief Test whether the type has species
 * 
 * @param handle populated handle
 * @param flag flag signifying whether the type has species
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_hasSpecies(struct MxParticleTypeHandle *handle, bool *flag);

/**
 * @brief Get the type species. Fails if the type does not have species
 * 
 * @param handle populated handle
 * @param slist species list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getSpecies(struct MxParticleTypeHandle *handle, struct MxSpeciesListHandle *slist);

/**
 * @brief Set the type species. 
 * 
 * @param handle populated handle
 * @param slit species list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setSpecies(struct MxParticleTypeHandle *handle, struct MxSpeciesListHandle *slist);

/**
 * @brief Get the type mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getMass(struct MxParticleTypeHandle *handle, double *mass);

/**
 * @brief Set the type mass
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setMass(struct MxParticleTypeHandle *handle, double mass);

/**
 * @brief Get the type radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getRadius(struct MxParticleTypeHandle *handle, double *radius);

/**
 * @brief Set the type radius
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setRadius(struct MxParticleTypeHandle *handle, double radius);

/**
 * @brief Get the kinetic energy of all particles of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getKineticEnergy(struct MxParticleTypeHandle *handle, double *kinetic_energy);

/**
 * @brief Get the potential energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getPotentialEnergy(struct MxParticleTypeHandle *handle, double *potential_energy);

/**
 * @brief Get the target energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getTargetEnergy(struct MxParticleTypeHandle *handle, double *target_energy);

/**
 * @brief Set the target energy of all particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setTargetEnergy(struct MxParticleTypeHandle *handle, double target_energy);

/**
 * @brief Get the default minimum radius of this type. 
 * 
 * If a split event occurs, resulting particles will have a radius 
 * at least as great as this value. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getMinimumRadius(struct MxParticleTypeHandle *handle, double *minimum_radius);

/**
 * @brief Set the default minimum radius of this type. 
 * 
 * If a split event occurs, resulting particles will have a radius 
 * at least as great as this value. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setMinimumRadius(struct MxParticleTypeHandle *handle, double minimum_radius);

/**
 * @brief Get the default dynamics of particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getDynamics(struct MxParticleTypeHandle *handle, unsigned char *dynamics);

/**
 * @brief Set the default dynamics of particles of this type. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setDynamics(struct MxParticleTypeHandle *handle, unsigned char dynamics);

/**
 * @brief Get the number of particles that are a member of this type.
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getNumParticles(struct MxParticleTypeHandle *handle, int *numParts);

/**
 * @brief Get the i'th particle that's a member of this type.
 * 
 * @param handle populated handle
 * @param i index of particle to get
 * @param phandle particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getParticle(struct MxParticleTypeHandle *handle, int i, struct MxParticleHandleHandle *phandle);

/**
 * @brief Test whether this type is a cluster type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_isCluster(struct MxParticleTypeHandle *handle, bool *isCluster);

/**
 * @brief Convert the type to a cluster; fails if the type is not a cluster
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_toCluster(struct MxParticleTypeHandle *handle, struct MxClusterParticleTypeHandle *chandle);

/**
 * @brief Particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_createParticle(struct MxParticleTypeHandle *handle, int *pid, float *position, float *velocity);

/**
 * @brief Particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param str JSON string defining a particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_createParticleS(struct MxParticleTypeHandle *handle, int *pid, const char *str);

/**
 * @brief Particle type constructor. 
 * 
 * New type is constructed from the definition of the calling type. 
 * 
 * @param handle populated handle
 * @param _name name of the new type
 * @param newTypehandle handle to populate with new type
 * @return MxParticleType* 
 */
CAPI_FUNC(HRESULT) MxCParticleType_newType(struct MxParticleTypeHandle *handle, const char *_name, struct MxParticleTypeHandle *newTypehandle);

/**
 * @brief Registers a type with the engine.
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_registerType(struct MxParticleTypeHandle *handle);

/**
 * @brief Tests whether this type is registered
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_isRegistered(struct MxParticleTypeHandle *handle, bool *isRegistered);

/**
 * @brief Test whether this type is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getFrozen(struct MxParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setFrozen(struct MxParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getFrozenX(struct MxParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the x-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setFrozenX(struct MxParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getFrozenY(struct MxParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the y-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setFrozenY(struct MxParticleTypeHandle *handle, bool frozen);

/**
 * @brief Test whether this type is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getFrozenZ(struct MxParticleTypeHandle *handle, bool *frozen);

/**
 * @brief Set whether this type is frozen along the z-direction
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setFrozenZ(struct MxParticleTypeHandle *handle, bool frozen);

/**
 * @brief Get the temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getTemperature(struct MxParticleTypeHandle *handle, double *temperature);

/**
 * @brief Get the target temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getTargetTemperature(struct MxParticleTypeHandle *handle, double *temperature);

/**
 * @brief Set the target temperature of this type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_setTargetTemperature(struct MxParticleTypeHandle *handle, double temperature);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_toString(struct MxParticleTypeHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * The returned type is automatically registered with the engine. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_fromString(struct MxParticleTypeHandle *handle, const char *str);


////////////////////
// MxParticleList //
////////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_init(struct MxParticleListHandle *handle);

/**
 * @brief Initialize an instance with an array of particles
 * 
 * @param handle handle to populate
 * @param particles particles to put in the list
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_initP(struct MxParticleListHandle *handle, struct MxParticleHandleHandle **particles, unsigned int numParts);

/**
 * @brief Initialize an instance with an array of particle ids
 * 
 * @param handle handle to populate
 * @param parts particle ids to put in the list
 * @param numParts number of particle ids
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_initI(struct MxParticleListHandle *handle, int *parts, unsigned int numParts);

/**
 * @brief Copy an instance
 * 
 * @param source list to copy
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_copy(struct MxParticleListHandle *source, struct MxParticleListHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_destroy(struct MxParticleListHandle *handle);

/**
 * @brief Get the particle ids in the list
 * 
 * @param handle populated handle
 * @param parts particle id array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getIds(struct MxParticleListHandle *handle, int **parts);

/**
 * @brief Get the number of particles
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getNumParts(struct MxParticleListHandle *handle, unsigned int *numParts);

/**
 * @brief Free the memory associated with the parts list.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_free(struct MxParticleListHandle *handle);

/**
 * @brief Insert the given id into the list, returns the index of the item. 
 * 
 * @param handle populated handle
 * @param item id to insert
 * @param index index of the particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_insertI(struct MxParticleListHandle *handle, int item, unsigned int *index);

/**
 * @brief Inserts the given particle into the list, returns the index of the particle. 
 * 
 * @param handle populated handle
 * @param particle particle to insert
 * @param index index of the particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_insertP(struct MxParticleListHandle *handle, struct MxParticleHandleHandle *particle, unsigned int *index);

/**
 * @brief Looks for the item with the given id and deletes it from the list
 * 
 * @param id id to remove
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_remove(struct MxParticleListHandle *handle, int id);

/**
 * @brief inserts the contents of another list
 * 
 * @param handle populated handle
 * @param other another list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_extend(struct MxParticleListHandle *handle, struct MxParticleListHandle *other);

/**
 * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
 * 
 * @param handle populated handle
 * @param i index of lookup
 * @param item returned item if found
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_item(struct MxParticleListHandle *handle, unsigned int i, struct MxParticleHandleHandle *item);

/**
 * @brief Initialize an instance populated with all current particles
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getAll(struct MxParticleListHandle *handle);

/**
 * @brief Get the virial tensor of the particles
 * 
 * @param handle populated handle
 * @param virial 9-element allocated array, virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getVirial(struct MxParticleListHandle *handle, float **virial);

/**
 * @brief Get the radius of gyration of the particles
 * 
 * @param handle populated handle
 * @param rog radius of gyration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getRadiusOfGyration(struct MxParticleListHandle *handle, float *rog);

/**
 * @brief Get the center of mass of the particles
 * 
 * @param handle populated handle
 * @param com 3-element allocated array, center of mass
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getCenterOfMass(struct MxParticleListHandle *handle, float **com);

/**
 * @brief Get the centroid of the particles
 * 
 * @param handle populated handle
 * @param cent 3-element allocated array, centroid
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getCentroid(struct MxParticleListHandle *handle, float **cent);

/**
 * @brief Get the moment of inertia of the particles
 * 
 * @param handle populated handle
 * @param moi 9-element allocated array, moment of inertia
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getMomentOfInertia(struct MxParticleListHandle *handle, float **moi);

/**
 * @brief Get the particle positions
 * 
 * @param handle populated handle
 * @param positions array of 3-element arrays, positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getPositions(struct MxParticleListHandle *handle, float **positions);

/**
 * @brief Get the particle velocities
 * 
 * @param handle populated handle
 * @param velocities array of 3-element arrays, velocities; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getVelocities(struct MxParticleListHandle *handle, float **velocities);

/**
 * @brief Get the forces acting on the particles
 * 
 * @param handle populated handle
 * @param forces array of 3-element arrays, forces; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_getForces(struct MxParticleListHandle *handle, float **forces);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system at the center of the universe
 * 
 * @param handle populated handle
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_sphericalPositions(struct MxParticleListHandle *handle, float **coordinates);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system with a specified origin
 * 
 * @param handle populated handle
 * @param origin optional origin of coordinates; default is center of universe
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_sphericalPositionsO(struct MxParticleListHandle *handle, float *origin, float **coordinates);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleList_toString(struct MxParticleListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleList_fromString(struct MxParticleListHandle *handle, const char *str);


////////////////////////
// MxParticleTypeList //
////////////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_init(struct MxParticleTypeListHandle *handle);

/**
 * @brief Initialize an instance with an array of particle types
 * 
 * @param handle handle to populate
 * @param parts particle types to put in the list
 * @param numParts number of particle types
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_initP(struct MxParticleTypeListHandle *handle, struct MxParticleTypeHandle **parts, unsigned int numParts);

/**
 * @brief Initialize an instance with an array of particle type ids
 * 
 * @param handle handle to populate
 * @param parts particle type ids to put in the list
 * @param numParts number of particle type ids
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_initI(struct MxParticleTypeListHandle *handle, int *parts, unsigned int numParts);

/**
 * @brief Copy an instance
 * 
 * @param source list to copy
 * @param destination handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_copy(struct MxParticleTypeListHandle *source, struct MxParticleTypeListHandle *destination);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_destroy(struct MxParticleTypeListHandle *handle);

/**
 * @brief Get the particle type ids in the list
 * 
 * @param handle populated handle
 * @param parts particle type id array
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getIds(struct MxParticleTypeListHandle *handle, int **parts);

/**
 * @brief Get the number of particle types
 * 
 * @param handle populated handle
 * @param numParts number of particle types
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getNumParts(struct MxParticleTypeListHandle *handle, unsigned int *numParts);

/**
 * @brief Free the memory associated with the list.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_free(struct MxParticleTypeListHandle *handle);

/**
 * @brief Insert the given id into the list, returns the index of the item. 
 * 
 * @param handle populated handle
 * @param item id to insert
 * @param index index of the particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_insertI(struct MxParticleTypeListHandle *handle, int item, unsigned int *index);

/**
 * @brief Inserts the given particle type into the list, returns the index of the particle. 
 * 
 * @param handle populated handle
 * @param ptype particle type to insert
 * @param index index of the particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_insertP(struct MxParticleTypeListHandle *handle, struct MxParticleTypeHandle *ptype, unsigned int *index);

/**
 * @brief Looks for the item with the given id and deletes it from the list
 * 
 * @param id id to remove
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_remove(struct MxParticleTypeListHandle *handle, int id);

/**
 * @brief inserts the contents of another list
 * 
 * @param handle populated handle
 * @param other another list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_extend(struct MxParticleTypeListHandle *handle, struct MxParticleTypeListHandle *other);

/**
 * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
 * 
 * @param handle populated handle
 * @param i index of lookup
 * @param item returned item if found
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_item(struct MxParticleTypeListHandle *handle, unsigned int i, struct MxParticleTypeHandle *item);

/**
 * @brief Initialize an instance populated with all current particles
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getAll(struct MxParticleTypeListHandle *handle);

/**
 * @brief Get the virial tensor of the particles
 * 
 * @param handle populated handle
 * @param virial 9-element allocated array, virial tensor
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getVirial(struct MxParticleTypeListHandle *handle, float *virial);

/**
 * @brief Get the radius of gyration of the particles
 * 
 * @param handle populated handle
 * @param rog radius of gyration
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getRadiusOfGyration(struct MxParticleTypeListHandle *handle, float *rog);

/**
 * @brief Get the center of mass of the particles
 * 
 * @param handle populated handle
 * @param com 3-element allocated array, center of mass
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getCenterOfMass(struct MxParticleTypeListHandle *handle, float **com);

/**
 * @brief Get the centroid of the particles
 * 
 * @param handle populated handle
 * @param cent 3-element allocated array, centroid
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getCentroid(struct MxParticleTypeListHandle *handle, float **cent);

/**
 * @brief Get the moment of inertia of the particles
 * 
 * @param handle populated handle
 * @param moi 9-element allocated array, moment of inertia
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getMomentOfInertia(struct MxParticleTypeListHandle *handle, float **moi);

/**
 * @brief Get the particle positions
 * 
 * @param handle populated handle
 * @param positions array of 3-element arrays, positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getPositions(struct MxParticleTypeListHandle *handle, float **positions);

/**
 * @brief Get the particle velocities
 * 
 * @param handle populated handle
 * @param velocities array of 3-element arrays, velocities; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getVelocities(struct MxParticleTypeListHandle *handle, float **velocities);

/**
 * @brief Get the forces acting on the particles
 * 
 * @param handle populated handle
 * @param forces array of 3-element arrays, forces; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getForces(struct MxParticleTypeListHandle *handle, float **forces);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system at the center of the universe
 * 
 * @param handle populated handle
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_sphericalPositions(struct MxParticleTypeListHandle *handle, float **coordinates);

/**
 * @brief Get the spherical coordinates of each particle for a coordinate system with a specified origin
 * 
 * @param handle populated handle
 * @param origin optional origin of coordinates; default is center of universe
 * @param coordinates array of 3-element arrays, spherical positions; order is according to the ordering of the list
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_sphericalPositionsO(struct MxParticleTypeListHandle *handle, float *origin, float **coordinates);

/**
 * @brief Get a particle list populated with particles of all current particle types
 * 
 * @param handle populated handle
 * @param plist handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_getParticles(struct MxParticleTypeListHandle *handle, struct MxParticleListHandle *plist);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation; can be used as an argument in a type particle factory
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_toString(struct MxParticleTypeListHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCParticleTypeList_fromString(struct MxParticleTypeListHandle *handle, const char *str);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Iterates over all parts, does a verify
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticle_Verify();

/**
 * @brief Get a registered particle type by type name
 * 
 * @param handle handle to populate
 * @param name name of particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_FindFromName(struct MxParticleTypeHandle *handle, const char* name);

/**
 * @brief Get a registered particle type by type id
 * 
 * @param handle handle to populate
 * @param name name of particle type
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCParticleType_getFromId(struct MxParticleTypeHandle *handle, unsigned int pid);

/**
 * @brief Get an array of available particle type colors
 * 
 * @return unsigned int *
 */
CAPI_FUNC(unsigned int*) MxCParticle_Colors();

#endif // _WRAPS_C_MXCPARTICLE_H_