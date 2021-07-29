/*
 * MxForces.h
 *
 *  Created on: Sep 19, 2018
 *      Author: andy
 */

#ifndef SRC_MXFORCES_H_
#define SRC_MXFORCES_H_

#include "mechanica_private.h"
#include <vector>

// A propagator using these to determine what information to give a force
enum FORCABLE_TYPE {
    FORCABLE_NONE, 
    FORCABLE_CELL, 
    FORCABLE_POLYGON
};

struct MxForcable;

/**
 * Interface that force objects implement.
 *
 * Forces can be time-dependent, and contain state variables. Forces can be stepped
 * in time just like physical objects.
 */
struct IForce {

    virtual FORCABLE_TYPE dataType() { return FORCABLE_TYPE::FORCABLE_NONE; }

    /**
     * Called when the main time step changes.
     */
    virtual HRESULT setTime(float time) = 0;

    /**
     * Apply forces to a set of objects.
     */
    virtual HRESULT applyForce(float time, const std::vector<MxForcable*> &objs) const = 0;
};

struct MxForcableType {

    std::vector<IForce*> forces;

};

struct MxForcable {

    MxForcableType *forceType;

    MxForcable(MxForcableType *type) : forceType(type) {}

};


struct MxForces
{
};

#endif /* SRC_MXFORCES_H_ */
