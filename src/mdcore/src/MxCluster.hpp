/*
 * MxCluster.h
 *
 *  Created on: Aug 28, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXCLUSTER_H_
#define SRC_MDCORE_SRC_MXCLUSTER_H_

#include <MxParticle.h>

/**
 * @brief The cluster analogue to :class:`MxParticle`. 
 * 
 */
struct MxCluster : MxParticle
{
};

struct MxClusterParticleHandle;

/**
 * @brief The cluster analogue to :class:`MxParticleType`. 
 * 
 */
struct MxClusterParticleType : MxParticleType {

    MxClusterParticleType(const bool &noReg=false);

    /**
     * @brief Tests where this cluster has a particle type
     * 
     * @param type type to test
     * @return true if this cluster has the type
     */
    bool hasType(const MxParticleType *type);

    /**
     * @brief Registers a type with the engine. 
     * 
     * Note that this does not occur automatically for basic usage, 
     * to allow type-specific operations independently of the engine. 
     * 
     * Also registers all unregistered constituent types. 
     * 
     * @return HRESULT 
     */
    HRESULT registerType();

    /**
     * @brief Get the type engine instance
     * 
     * @return MxClusterParticleType* 
     */
    virtual MxClusterParticleType *get();

};

/**
 * @brief The cluster analogue to :class:`MxParticleHandle`. 
 * 
 * These are special in that they can create particles of their 
 * constituent particle types, much like a :class:`MxParticleType`. 
 * 
 */
struct MxClusterParticleHandle : MxParticleHandle {
    MxClusterParticleHandle();
    MxClusterParticleHandle(const int &id, const int &typeId);

    /**
     * @brief Constituent particle constructor. 
     * 
     * The created particle will belong to this cluster. 
     * 
     * @param partType type of particle to create
     * @param position position of new particle, optional
     * @param velocity velocity of new particle, optional
     * @return MxParticleHandle* 
     */
    MxParticleHandle *operator()(MxParticleType *partType, 
                                 MxVector3f *position=NULL, 
                                 MxVector3f *velocity=NULL);

    MxParticleHandle* fission(MxVector3f *axis=NULL, 
                              bool *random=NULL, 
                              float *time=NULL, 
                              MxVector3f *normal=NULL, 
                              MxVector3f *point=NULL);

    /**
     * @brief Split the cluster. 
     * 
     * @param axis axis of split, optional
     * @param random divide by randomly and evenly allocating constituent particles, optional
     * @param time time at which to implement the split; currently not supported
     * @param normal normal vector of cleavage plane, optional
     * @param point point on cleavage plane, optional
     * @return MxParticleHandle* 
     */
    MxParticleHandle* split(MxVector3f *axis=NULL, 
                            bool *random=NULL, 
                            float *time=NULL, 
                            MxVector3f *normal=NULL, 
                            MxVector3f *point=NULL);
    
    float getRadiusOfGyration();
    MxVector3f getCenterOfMass();
    MxVector3f getCentroid();
    MxMatrix3f getMomentOfInertia();
};

/**
 * adds an existing particle to the cluster.
 */
CAPI_FUNC(int) MxCluster_AddParticle(struct MxCluster *cluster, struct MxParticle *part);


/**
 * Computes the aggregate quanties such as total mass, position, acceleration, etc...
 * from the contained particles. 
 */
CAPI_FUNC(int) MxCluster_ComputeAggregateQuantities(struct MxCluster *cluster);

/**
 * creates a new particle, and adds it to the cluster.
 */
CAPI_FUNC(MxParticle*) MxCluster_CreateParticle(MxCluster *cluster,
                                                MxParticleType* particleType, 
                                                MxVector3f *position=NULL, 
                                                MxVector3f *velocity=NULL);

/**
 * @brief Get a registered cluster type by type name
 * 
 * @param name name of cluster type
 * @return MxClusterParticleType* 
 */
CAPI_FUNC(MxClusterParticleType*) MxClusterParticleType_FindFromName(const char* name);

/**
 * internal function to initalize the particle and particle types
 */
HRESULT _MxCluster_init();


#endif /* SRC_MDCORE_SRC_MXCLUSTER_H_ */
