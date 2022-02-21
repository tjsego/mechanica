/*
 * MxBody.h
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_BODY_H_
#define SRC_MDCORE_BODY_H_

#include <platform.h>
#include <MxParticle.h>
#include "../../types/mx_types.h"
#include <Magnum/Math/Range.h>

struct MxBodyHandle;

struct MxBody
{
    /** Particle position */
    union {
        MX_ALIGNED(FPTYPE, 16) x[4];
        MX_ALIGNED(MxVector3f, 16) position;

        struct {
            float __dummy[3];
            uint32_t creation_time;
        };
    };

    /** linear velocity */
    union {
        MX_ALIGNED(FPTYPE, 16) v[4];
        MX_ALIGNED(MxVector3f, 16) velocity;
    };
    
    /**
     * linear force
     *
     * ONLY the coherent part of the force should go here. We use multi-step
     * integrators, that need to separate the random and coherent forces.
     */
    union {
        MX_ALIGNED(FPTYPE, 16) f[4];
        MX_ALIGNED(MxVector3f, 16) force;
    };
    
    union {
        MX_ALIGNED(FPTYPE, 16) pad_orientation[4];
        MX_ALIGNED(MxQuaternionf, 16) orientation;
    };
    
    /** angular velocity */
    union {
        MX_ALIGNED(FPTYPE, 16) _spin[4];
        MX_ALIGNED(MxVector3f, 16) spin;
    };

    union {
        MX_ALIGNED(FPTYPE, 16) _torque[4];
        MX_ALIGNED(MxVector3f, 16) torque;
    };
    
    /**
     * inverse rotation transform. 
     */
    MxQuaternionf inv_orientation;
    
    /**
     * update the aabb on motion. 
     */
    MxVector3f aabb_min_size;
    
    Magnum::Range3D aabb;
        

    /** random force force */
    union {
        MX_ALIGNED(MxVector3f, 16) persistent_force;
    };

    // inverse mass
    double imass;

    double mass;
    
    // index of the object in some array, negative for invalid.
    int32_t id;

    /** Particle flags */
    uint32_t flags;

    /**
     * pointer to the python 'wrapper'. Need this because the particle data
     * gets moved around between cells, and python can't hold onto that directly,
     * so keep a pointer to the python object, and update that pointer
     * when this object gets moved.
     *
     * initialzied to null, and only set when .
     */
    MxBodyHandle *_handle;

    /**
     * public way of getting the pyparticle. Creates and caches one if
     * it's not there. Returns a inc-reffed handle, caller is responsible
     * for freeing it.
     */
    MxBodyHandle *handle();


    // style pointer, set at object construction time.
    // may be re-set by users later.
    // the base particle type has a default style.
    MxStyle *style;

    /**
     * pointer to state vector (optional)
     */
    struct MxStateVector *state_vector;
    
    MxBody();
};

struct MxBodyHandle
{
    int32_t id;

    MxBody *body();

    MxBodyHandle() : id(0) {}
    MxBodyHandle(const int32_t &id) : id(id) {}

    HRESULT move(const MxVector3f &by);
    HRESULT move(const std::vector<float> &by);

    HRESULT rotate(const MxVector3f &by);
    HRESULT rotate(const std::vector<float> &by);

    MxVector3f getPosition() const;
    void setPosition(const MxVector3f &position);
    void setPosition(const std::vector<float> &position);

    MxVector3f getVelocity() const;
    void setVelocity(const MxVector3f &velocity);
    void setVelocity(const std::vector<float> &velocity);
    
    MxVector3f getForce() const;
    void setForce(const MxVector3f &force);
    void setForce(const std::vector<float> &force);

    MxQuaternionf getOrientation() const;
    void setOrientation(const MxQuaternionf &orientation);
    void setOrientation(const std::vector<float> &orientation);
    
    MxVector3f getSpin() const;
    void setSpin(const MxVector3f &spin);
    void setSpin(const std::vector<float> &spin);

    MxVector3f getTorque() const;
    void setTorque(const MxVector3f &torque);
    void setTorque(const std::vector<float> &torque);
    
    MxStateVector *getSpecies() const;
};

#endif /* SRC_MDCORE_BODY_H_ */
