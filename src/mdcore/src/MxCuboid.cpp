/*
 * MxCuboid.cpp
 *
 *  Created on: Jan 17, 2021
 *      Author: andy
 */

#include <MxCuboid.hpp>
#include <engine.h>
#include <cuboid_eval.hpp>
#include <../../mx_error.h>

#define CUBOID_SELF(handle) \
    MxCuboid *self = &_Engine.s.cuboids[((MxCuboidHandle*)handle)->id]; \
    if(self == NULL) { \
        PyErr_SetString(PyExc_ReferenceError, "Cuboid has been destroyed or is invalid"); \
    }

#define CUBOID_PROP_SELF(handle) \
    MxCuboid *self = &_Engine.s.cuboids[((MxCuboidHandle*)handle)->id]; \
    if(self == NULL) { \
        PyErr_SetString(PyExc_ReferenceError, "Cuboid has been destroyed or is invalid"); \
    }


MxCuboid::MxCuboid() {
    bzero(this, sizeof(MxCuboid));
    orientation = MxQuaternionf();
}

MxCuboidHandle *MxCuboid::create(MxVector3f *pos, MxVector3f *size, MxVector3f *orientation) {
    auto _pos = pos ? *pos : engine_center();
    auto _size = size ? *size : MxVector3f(1.0, 1.0, 1.0);
    auto _orientation = orientation ? *orientation : MxVector3f(0.0, 0.0, 0.0);
    return new MxCuboidHandle(_pos, _size, _orientation);
}

MxCuboid *MxCuboidHandle::cuboid() {
    try {
        CUBOID_SELF(this);
        return self;
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxCuboidHandle::MxCuboidHandle(const MxVector3f &pos, const MxVector3f &size, const MxVector3f &orientation) : 
    MxBodyHandle()
{
    init(pos, size, orientation);
}

void MxCuboidHandle::init(const MxVector3f &pos, const MxVector3f &size, const MxVector3f &orientation) {
    if(id > 0) return;

    try {
        MxCuboid c;
        
        HRESULT err;
        
        MxCuboid *p;
        
        c.position = pos;
        c.size = size;
        
        MxVector3f angle = orientation;
        
        MxQuaternionf qx = MxQuaternionf::rotation(angle[0], MxVector3f::xAxis());
        MxQuaternionf qy = MxQuaternionf::rotation(angle[1], MxVector3f::yAxis());
        MxQuaternionf qz = MxQuaternionf::rotation(angle[2], MxVector3f::zAxis());
        
        c.orientation = qx * qy * qz;
        
        MxCuboid_UpdateAABB(&c);
        
        if(!SUCCEEDED((err = engine_addcuboid(&_Engine, &c, &p)))) {
            throw std::runtime_error("Failed to add cuboid");
        }
        
        p->_handle = this;

        this->id = p->id;

    }
    catch (const std::exception &e) {
        mx_exp(e);
    }
}

void MxCuboidHandle::scale(const MxVector3f &scale) {
    try {
        CUBOID_SELF(this);
        self->size = MxMatrix4f::scaling(scale).transformVector(self->size);
    }
    catch(const std::exception &e) {
        mx_exp(e);
    }
}

void MxCuboid_UpdateAABB(MxCuboid *c) {
    MxVector3f min = MxVector3f(std::numeric_limits<float>::max());
    MxVector3f max = MxVector3f(std::numeric_limits<float>::min());
    MxVector3f halfSize = 0.5 * c->size;
    MxVector3f pos = c->position;
    MxVector3f points[] =  {
        c->orientation.transformVector({ halfSize[0],  halfSize[1],  halfSize[2]}) + pos,
        c->orientation.transformVector({ halfSize[0], -halfSize[1],  halfSize[2]}) + pos,
        c->orientation.transformVector({-halfSize[0], -halfSize[1],  halfSize[2]}) + pos,
        c->orientation.transformVector({-halfSize[0],  halfSize[1],  halfSize[2]}) + pos,
        
        c->orientation.transformVector({ halfSize[0],  halfSize[1],  -halfSize[2]}) + pos,
        c->orientation.transformVector({ halfSize[0], -halfSize[1],  -halfSize[2]}) + pos,
        c->orientation.transformVector({-halfSize[0], -halfSize[1],  -halfSize[2]}) + pos,
        c->orientation.transformVector({-halfSize[0],  halfSize[1],  -halfSize[2]}) + pos,
    };
    
    for(int i = 0; i < 8; ++i) {
        min = {std::min(min[0], points[i][0]), std::min(min[1], points[i][1]), std::min(min[2], points[i][2])};
        max = {std::max(max[0], points[i][0]), std::max(max[1], points[i][1]), std::max(max[2], points[i][2])};
    }
    
    c->aabb = {min, max};
    
    // TODO: only need to compute if rotation. 
    c->inv_orientation = c->orientation.inverted();
    
    MxVector3f p2 = c->orientation.transformVector(MxVector3f{1, 0, 1});
}
