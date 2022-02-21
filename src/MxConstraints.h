/*
 * MxConstraints.h
 *
 *  Created on: Sep 19, 2018
 *      Author: andy
 */

#ifndef SRC_MXCONSTRAINTS_H_
#define SRC_MXCONSTRAINTS_H_

#include "mechanica_private.h"
#include <vector>

// A propagator using these to determine what information to give a constraint
enum CONSTRAINABLE_TYPE {
    CONSTRAINABLE_NONE, 
    CONSTRAINABLE_CELL, 
    CONSTRAINABLE_POLYGON
};

struct MxConstrainable;

struct IConstraint {

    virtual CONSTRAINABLE_TYPE dataType() { return CONSTRAINABLE_TYPE::CONSTRAINABLE_NONE; }

    virtual HRESULT setTime(float time) = 0;

    virtual float energy(const std::vector<MxConstrainable*> &objs) = 0;

    virtual HRESULT project(const std::vector<MxConstrainable*> &obj) = 0;
};


struct MxConstrainableType {

    std::vector<IConstraint*> constraints;

};

struct MxConstrainable {

    MxConstrainableType *constraintType;

    MxConstrainable(MxConstrainableType *type) : constraintType(type) {}

};


class MxConstraints
{
};

#endif /* SRC_MXCONSTRAINTS_H_ */
