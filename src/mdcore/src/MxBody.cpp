/*
 * MxBody.cpp
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */

#include <MxBody.hpp>
#include <../../mx_error.h>
#include <engine.h>

// TODO: cheap, quick hack, move this to function pointer or somethign...
#include <MxCuboid.hpp>

#define update_aabb(body) MxCuboid_UpdateAABB((MxCuboid*)body)


#define BODY_SELF(handle) \
    MxBody *self = &_Engine.s.cuboids[handle->id]; \
    if(self == NULL) { \
        throw std::runtime_error("Body has been destroyed or is invalid"); \
    }

#define BODY_PROP_SELF(handle) \
    MxBody *self = &_Engine.s.cuboids[handle->id]; \
    if(self == NULL) { \
        throw std::runtime_error("Cuboid has been destroyed or is invalid"); \
        return -1; \
    }

MxBodyHandle *MxBody::handle() {
    if(!_handle) _handle = new MxBodyHandle(this->id);

    return _handle;
}

MxBody::MxBody() {
    bzero(this, sizeof(MxBody));
    orientation = MxQuaternionf();
}

static HRESULT body_move(MxBodyHandle *handle, const MxVector3f &by) {
    try {
        BODY_SELF(handle);
        
        self->position += by;
        
        update_aabb(self);

        return 1;
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxBodyHandle::move(const MxVector3f &by) {
    if(body_move(this, by) == 0) return 1;
    return 0;
}

HRESULT MxBodyHandle::move(const std::vector<float> &by) {
    return move(MxVector3f(by));
}

HRESULT body_rotate(MxBodyHandle *handle, const MxVector3f &by) {
    try {
        BODY_SELF(handle);
        
        MxQuaternionf qx = MxQuaternionf::rotation(by[0], MxVector3f::xAxis());
        MxQuaternionf qy = MxQuaternionf::rotation(by[1], MxVector3f::yAxis());
        MxQuaternionf qz = MxQuaternionf::rotation(by[2], MxVector3f::zAxis());
        
        MxQuaternionf test = MxQuaternionf(by).normalized();
        
        MxQuaternionf t2 = qx * qy * qz;
        
        self->orientation = self->orientation * qx * qy * qz;
        
        update_aabb(self);
        
        return 1;
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxBodyHandle::rotate(const MxVector3f &by) {
    if(body_rotate(this, by) == 0) return 1;
    return 0;
}

HRESULT MxBodyHandle::rotate(const std::vector<float> &by) {
    return rotate(MxVector3f(by));
}

MxBody *MxBodyHandle::body() {
    try {
        BODY_SELF(this);
        return self;
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

// todo: generate MxBodyHandle python property: position
MxVector3f MxBodyHandle::getPosition() const {
    BODY_SELF(this);
    return self->position;
}

void MxBodyHandle::setPosition(const MxVector3f &position) {
    try {
        BODY_SELF(this);
        self->position = position;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setPosition(const std::vector<float> &position) {
    return setPosition(MxVector3f(position));
}

// todo: generate MxBodyHandle python property: velocity
MxVector3f MxBodyHandle::getVelocity() const {
    BODY_SELF(this);
    return self->velocity;
}

void MxBodyHandle::setVelocity(const MxVector3f &velocity) {
    try {
        BODY_SELF(this);
        self->velocity = velocity;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setVelocity(const std::vector<float> &velocity) {
    return setVelocity(MxVector3f(velocity));
}

// todo: generate MxBodyHandle python property: force
MxVector3f MxBodyHandle::getForce() const {
    BODY_SELF(this);
    return self->force;
}

void MxBodyHandle::setForce(const MxVector3f &force) {
    try {
        BODY_SELF(this);
        self->force = force;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setForce(const std::vector<float> &force) {
    return setForce(MxVector3f(force));
}

MxQuaternionf MxBodyHandle::getOrientation() const {
    BODY_SELF(this);
    return self->orientation;
}

void MxBodyHandle::setOrientation(const MxQuaternionf &orientation) {
    try {
        BODY_SELF(this);
        self->orientation = orientation;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setOrientation(const std::vector<float> &orientation) {
    return setOrientation(MxQuaternionf(orientation));
}

MxVector3f MxBodyHandle::getSpin() const {
    BODY_SELF(this);
    return self->spin;
}

void MxBodyHandle::setSpin(const MxVector3f &spin) {
    try {
        BODY_SELF(this);
        self->spin = spin;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setSpin(const std::vector<float> &spin) {
    return setSpin(MxVector3f(spin));
}

MxVector3f MxBodyHandle::getTorque() const {
    BODY_SELF(this);
    return self->torque;
}

void MxBodyHandle::setTorque(const MxVector3f &torque) {
    try {
        BODY_SELF(this);
        self->torque = torque;
    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxBodyHandle::setTorque(const std::vector<float> &torque) {
    return setTorque(MxVector3f(torque));
}


// todo: generate MxBodyHandle python property: species
MxStateVector *MxBodyHandle::getSpecies() const {
    BODY_SELF(this);
    return self->state_vector;
}
