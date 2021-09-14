/*
 * MxCluster.cpp
 *
 *  Created on: Aug 28, 2020
 *      Author: andy
 */

#include <MxCluster.hpp>

/* include some standard header files */
#include <stdlib.h>
#include <math.h>
#include <MxParticle.h>
#include "fptype.h"
#include <iostream>
#include "../../MxUtil.h"
#include "../../MxLogger.h"
#include "../../mx_error.h"

// python type info
#include <structmember.h>
#include <MxPy.h>
#include "engine.h"
#include "space.h"
#include "mx_runtime.h"
#include "space_cell.h"

#include "../../rendering/NOMStyle.hpp"

#include <metrics.h>

/**
 * removes a particle from the list at the index.
 * returns null if not found
 */
static MxParticle *remove_particle_at_index(MxCluster *cluster, int index) {
    if(index >= cluster->nr_parts) {
        return NULL;
    }
    
    int pid = cluster->parts[index];
    
    for(int i = index; i + 1 < cluster->nr_parts; ++i) {
        cluster->parts[i] = cluster->parts[i+i];
    }
    
    cluster->nr_parts -= 1;
    cluster->parts[cluster->nr_parts] = -1;
    
    MxParticle *part = _Engine.s.partlist[pid];
    
    part->clusterId = -1;
    
    return part;
}

static MxParticleHandle* cluster_fission_plane(MxParticle *cluster, const MxVector4f &plane) {
    
    Log(LOG_INFORMATION) << ", plane: " << plane;
    
    // particles to move to daughter cluster.
    // only perform a split if the contained particles can be split into
    // two non-empty sets.
    std::vector<int> dparts;
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        MxParticle *p = cluster->particle(i);
        float dist = plane.distanceScaled(p->global_position());
        
        //Log(LOG_DEBUG) << "particle[" << i << "] position: " << p->global_position() << ", dist: " << dist;
        
        if(dist < 0) {
            dparts.push_back(p->id);
        }
    }
    
    if(dparts.size() > 0 && dparts.size() < cluster->nr_parts) {
        
        MxParticleHandle *_daughter = MxParticle_New(cluster->_pyparticle->type(),  NULL,  NULL);
        MxCluster *daughter = (MxCluster*)MxParticle_Get(_daughter);
        assert(daughter);
        
        Log(LOG_TRACE) << "split cluster "
        << cluster->id << " into ("
        << cluster->id << ":" << (cluster->nr_parts - dparts.size())
        << ", "
        << daughter->id << ": " << dparts.size() << ")" << std::endl;
        
        for(int i = 0; i < dparts.size(); ++i) {
            cluster->removepart(dparts[i]);
            daughter->addpart(dparts[i]);
        }
        
        return _daughter;
    }
    else {
        return NULL;
    }
}

static MxParticleHandle* cluster_fission_normal_point(MxParticle *cluster,
    const MxVector3f &normal, const MxVector3f &point) {
    
    Log(LOG_DEBUG) << "normal: " << normal
                   << ", point: " << point << ", cluster center: "
                   << cluster->global_position();
    
    MxVector4f plane = MxVector4f::planeEquation(normal, point);
    
    return cluster_fission_plane(cluster, plane);
}


static MxParticleHandle* cluster_fission_axis(MxParticle *cluster,
    const MxVector3f &axis) {
    
    Log(LOG_DEBUG) << "axis: " << axis;
    
    MxVector3f p1 = cluster->global_position();
    
    MxVector3f p2 = p1 + axis;
    
    MxVector3f p3 = p1 + MxRandomUnitVector();
    
    MxVector4f plane = MxVector4f::planeEquation(p1, p2, p3);
    
    return cluster_fission_plane(cluster, plane);
}

int MxCluster_ComputeAggregateQuantities(struct MxCluster *cluster) {
    
    if(cluster->nr_parts <= 0) {
        return 0;
    }
    
    MxVector3f pos;
    
    // compute in global coordinates, particles can belong to different
    // space cells.
    /* Copy the position to x. */
    //for ( k = 0 ; k < 3 ; k++ )
    //    x[k] = s->partlist[id]->x[k] + s->celllist[id]
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[cluster->parts[i]];
        pos += p->global_position();
    }
    
    cluster->set_global_position(pos / cluster->nr_parts);
    
    return 0;
}


static MxParticleHandle* cluster_fission_random(MxParticle *cluster)
{
    MxParticleHandle *_daughter = MxParticle_New(cluster->_pyparticle->type(),  NULL,  NULL);
    
    MxCluster *daughter = (MxCluster*)MxParticle_Get(_daughter);
    assert(daughter);
    
    int halfIndex = cluster->nr_parts / 2;
    
    for(int i = halfIndex; i < cluster->nr_parts; ++i) {
        // adds to new cluster, sets id of contained particle.
        daughter->addpart(cluster->parts[i]);
        cluster->parts[i] = -1;
    }
    
    cluster->nr_parts = halfIndex;
    
    return _daughter;
}

MxClusterParticleType::MxClusterParticleType(const bool &noReg) : 
    MxParticleType(noReg) 
{
    this->particle_flags |= PARTICLE_CLUSTER;
}

bool MxClusterParticleType::hasType(const MxParticleType *type) {
    for (int tid = 0; tid < types.nr_parts; ++tid) {
        if (type->id == types.item(tid)->id) 
            return true;
    }
    return false;
}

// Registers a type with the engine. 
// Also registers all unregistered constituent types
HRESULT MxClusterParticleType::registerType() {
    for (int tid = 0; tid < types.nr_parts; ++tid) {
        auto type = types.item(tid);
        if (!type->isRegistered()) {
            auto result = type->registerType();
            if (result != S_OK) return result;
        }
    }

    return MxParticleType::registerType();
}

MxClusterParticleType *MxClusterParticleType::get() {
    return (MxClusterParticleType*)MxParticleType::get();
}

MxClusterParticleHandle::MxClusterParticleHandle() : 
    MxParticleHandle() 
{}

MxClusterParticleHandle::MxClusterParticleHandle(const int &id, const int &typeId) : 
    MxParticleHandle(id, typeId) 
{}

MxParticleHandle *MxClusterParticleHandle::operator()(MxParticleType *partType, 
                                                      MxVector3f *position, 
                                                      MxVector3f *velocity) 
{
    auto p = MxCluster_CreateParticle((MxCluster*)part(), partType, position, velocity);
    return new MxParticleHandle(p->id, p->typeId);
}

/**
 # split the cell with a cleavage plane, in normal/point form.
 split(normal=[x, y, z], point=[px, py, pz])
 
 # split the cell with a cleavage plane normal, but use the clusters center as the point
 split(normal=[x, y, z])
 
 # if no named arguments are given, split interprets the first argument as a cleavage normal:
 split([x, y, z])
 
 # split using a cleavage *axis*, here, the split will generate a cleavage plane
 # that contains the given axis. This method is designed for the epiboly project,
 # where you’d give it an axis that’s the vector between the yolk center, and the
 # center of the cell. This will split the cell perpendicular to the yolk
 split(axis=[x, y, z])
 
 # default version of split uses a random cleavage plane that intersects the
 # cell center
 split()
 
 # the old style, were randomly picks contained objects, and assigns half of them
 # to the daughter cell
 split(random=True)
*/
MxParticleHandle* MxClusterParticleHandle::fission(MxVector3f *axis, 
                                                   bool *random, 
                                                   float *time, 
                                                   MxVector3f *normal, 
                                                   MxVector3f *point) {

    Log(LOG_TRACE) ;
    
    auto *cluster = part();
    
    if(!(MxCluster*)cluster) {
        throw std::runtime_error("ERROR, given object is not a cluster");
        return NULL;
    }
    
    MxCluster_ComputeAggregateQuantities((MxCluster*)cluster);
    
    if(axis) {
        // use axis form of split
        return cluster_fission_axis(cluster, *axis);
    }
    
    if(random && *random) {
        // use random form of split
        return cluster_fission_random(cluster);
    }
    
    MxVector3f _normal;
    MxVector3f _point;
    
    // check if being called as an event, with the time number argument
    if(time) {
        float t = *time;
        Log(LOG_TRACE) << "cluster split event(cluster id: " << cluster->id
                  << ", count: " << cluster->nr_parts
                  << ", time: " << t << ")" << std::endl;
        _normal = MxRandomUnitVector();
        _point = cluster->global_position();
    }
    else {
        // normal documented usage
        _normal = normal ? *normal : MxRandomUnitVector();
        _point = point ? *point : cluster->global_position();
        
        Log(LOG_TRACE) << "using cleavage plane to split cluster" << std::endl;
    }
    
    return cluster_fission_normal_point(cluster, _normal, _point);
    // return cluster_fission(this, axis, random, time, normal, point);
}

MxParticleHandle* MxClusterParticleHandle::split(MxVector3f *axis, 
                                                 bool *random, 
                                                 float *time, 
                                                 MxVector3f *normal, 
                                                 MxVector3f *point) 
{ return fission(axis, random, time, normal, point); }

MxParticleList *MxClusterParticleHandle::items() {
    MxParticle *self = this->part();
    return new MxParticleList(self->nr_parts, self->parts);
}

float MxClusterParticleHandle::getRadiusOfGyration() {
    MxParticle *self = this->part();
    float result;
    MxParticles_RadiusOfGyration(self->parts, self->nr_parts, &result);
    return result;
}

MxVector3f MxClusterParticleHandle::getCenterOfMass() {
    MxParticle *self = this->part();
    MxVector3f result;
    MxParticles_CenterOfMass(self->parts, self->nr_parts, result.data());
    return result;
}

MxVector3f MxClusterParticleHandle::getCentroid() {
    MxParticle *self = this->part();
    MxVector3f result;
    MxParticles_CenterOfGeometry(self->parts, self->nr_parts, result.data());
    return result;
}

MxMatrix3f MxClusterParticleHandle::getMomentOfInertia() {
    MxParticle *self = this->part();
    MxMatrix3f result;
    MxParticles_MomentOfInertia(self->parts, self->nr_parts, result.data());
    return result;
}

/**
 * adds an existing particle to the cluster.
 */
int MxCluster_AddParticle(struct MxCluster *cluster, struct MxParticle *part) {
    cluster->addpart(part->id);
    return S_OK;
}

/**
 * creates a new particle, and adds it to the cluster.
 */
MxParticle *MxCluster_CreateParticle(MxCluster *cluster,
                                     MxParticleType* particleType, 
                                     MxVector3f *position, 
                                     MxVector3f *velocity)
{
    Log(LOG_TRACE);
    
    auto *type = &_Engine.types[cluster->typeId];
    if (!type->isCluster()) return NULL;
    
    auto *clusterType = (MxClusterParticleType*)type;
    Log(LOG_TRACE) << type->id << ", " << particleType->id << ", " << clusterType->hasType(particleType);
    if (!clusterType->hasType(particleType)) return NULL;

    auto handle = MxParticle_New(particleType, position, velocity, &cluster->clusterId);
    if (!handle) return NULL;

    auto particle = handle->part();
    if(MxCluster_AddParticle(cluster, particle) != S_OK) {
        handle->destroy();
        return NULL;
    }

    return particle;

}

MxClusterParticleType* MxClusterParticleType_FindFromName(const char* name) {
    MxParticleType *ptype = MxParticleType_FindFromName(name);
    if(!ptype) return NULL;
    return (MxClusterParticleType*)ptype;
}

HRESULT _MxCluster_init() {
    Log(LOG_TRACE);

    if(engine::nr_types != 1) {
        return mx_error(E_FAIL, "engine types already set, or not initialized in correct order");
    }

    auto type = new MxClusterParticleType();
    return S_OK;
}
