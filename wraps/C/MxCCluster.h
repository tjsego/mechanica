/**
 * @file MxCCluster.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxCluster
 * @date 2022-03-29
 */

#ifndef _WRAPS_C_MXCCLUSTER_H_
#define _WRAPS_C_MXCCLUSTER_H_

#include <mx_port.h>

#include "MxCParticle.h"

/**
 * @brief Cluster type definition in Mechanica C
 * 
 */
struct CAPI_EXPORT MxCClusterType {
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
    unsigned int numTypes;
    struct MxParticleTypeHandle **types;
};

// Handles

/**
 * @brief Handle to a @ref MxClusterParticleHandle instance
 * 
 */
struct CAPI_EXPORT MxClusterParticleHandleHandle {
    void *MxObj;
};

/**
 * @brief Handle to a @ref MxClusterParticleType instance
 * 
 */
struct CAPI_EXPORT MxClusterParticleTypeHandle {
    void *MxObj;
};


////////////////////
// MxCClusterType //
////////////////////


/**
 * @brief Get a default definition
 * 
 */
CAPI_FUNC(struct MxCClusterType) MxCClusterTypeDef_init();


/////////////////////////////
// MxClusterParticleHandle //
/////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param id particle id
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_init(struct MxClusterParticleHandleHandle *handle, int id);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_destroy(struct MxClusterParticleHandleHandle *handle);

/**
 * @brief Constituent particle constructor. 
 * 
 * The created particle will belong to this cluster. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @param handle populated handle
 * @param partType type of particle to create
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_createParticle(struct MxClusterParticleHandleHandle *handle, 
                                                           struct MxParticleTypeHandle *partTypeHandle, 
                                                           int *pid, 
                                                           float **position, 
                                                           float **velocity);

/**
 * @brief Constituent particle constructor. 
 * 
 * The created particle will belong to this cluster. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @param handle populated handle
 * @param partType type of particle to create
 * @param pid id of created particle
 * @param str JSON string defining a particle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_createParticleS(struct MxClusterParticleHandleHandle *handle, 
                                                            struct MxParticleTypeHandle *partTypeHandle, 
                                                            int *pid, 
                                                            const char *str);

/**
 * @brief Split the cluster along an axis. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param axis 3-component allocated axis of split
 * @param time time at which to implement the split; currently not supported
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_splitAxis(struct MxClusterParticleHandleHandle *handle, int *cid, float *axis, float time);

/**
 * @brief Split the cluster randomly. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param time time at which to implement the split; currently not supported
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_splitRand(struct MxClusterParticleHandleHandle *handle, int *cid, float time);

/**
 * @brief Split the cluster along a point and normal. 
 * 
 * @param handle populated handle
 * @param cid id of created cluster
 * @param time time at which to implement the split; currently not supported
 * @param normal 3-component normal vector of cleavage plane
 * @param point 3-component point on cleavage plane
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_split(struct MxClusterParticleHandleHandle *handle, int *cid, float time, float *normal, float *point);

/**
 * @brief Get the number of particles that are a member of this cluster.
 * 
 * @param handle populated handle
 * @param numParts number of particles
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getNumParts(struct MxClusterParticleHandleHandle *handle, int *numParts);

/**
 * @brief Get the i'th particle that's a member of this cluster.
 * 
 * @param handle populated handle
 * @param i index of particle to get
 * @param parthandle particle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getParticle(struct MxClusterParticleHandleHandle *handle, int i, struct MxParticleHandleHandle *parthandle);

/**
 * @brief Get the radius of gyration
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getRadiusOfGyration(struct MxClusterParticleHandleHandle *handle, float *radiusOfGyration);

/**
 * @brief Get the center of mass
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getCenterOfMass(struct MxClusterParticleHandleHandle *handle, float **com);

/**
 * @brief Get the centroid
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getCentroid(struct MxClusterParticleHandleHandle *handle, float **cent);

/**
 * @brief Get the moment of inertia
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleHandle_getMomentOfInertia(struct MxClusterParticleHandleHandle *handle, float **moi);


///////////////////////////
// MxClusterParticleType //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_init(struct MxClusterParticleTypeHandle *handle);

/**
 * @brief Initialize an instance from a definition
 * 
 * @param handle handle to populate
 * @param pdef definition
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_initD(struct MxClusterParticleTypeHandle *handle, struct MxCClusterType pdef);

/**
 * @brief Add a particle type to the types of a cluster
 * 
 * @param handle populated handle
 * @param phandle handle to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_addType(struct MxClusterParticleTypeHandle *handle, struct MxParticleTypeHandle *phandle);

/**
 * @brief Tests where this cluster has a particle type
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_hasType(struct MxClusterParticleTypeHandle *handle, struct MxParticleTypeHandle *phandle, bool *hasType);

/**
 * @brief Registers a type with the engine. 
 * 
 * Also registers all unregistered constituent types. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_registerType(struct MxClusterParticleTypeHandle *handle);

/**
 * @brief Cluster particle constructor.
 * 
 * @param handle populated handle
 * @param pid id of created particle
 * @param position pointer to 3-element array, or NULL for a random position
 * @param velocity pointer to 3-element array, or NULL for a random velocity
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_createParticle(struct MxClusterParticleTypeHandle *handle, int *pid, float *position, float *velocity);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get a registered cluster type by type name
 * 
 * @param handle handle to populate
 * @param name name of cluster type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_FindFromName(struct MxClusterParticleTypeHandle *handle, const char* name);

/**
 * @brief Get a registered cluster type by id
 * 
 * @param handle handle to populate
 * @param pid id of type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCClusterParticleType_getFromId(struct MxClusterParticleTypeHandle *handle, unsigned int pid);

#endif // _WRAPS_C_MXCCLUSTER_H_