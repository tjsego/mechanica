/*
 * MxForce.h
 *
 *  Created on: May 21, 2020
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_SRC_MXFORCE_H_
#define SRC_MDCORE_SRC_MXFORCE_H_

#include "platform.h"
#include "fptype.h"
#include "../../types/mx_types.h"

#include <limits>

enum MXFORCE_KIND {
    MXFORCE_ONEBODY,
    MXFORCE_PAIRWISE
};

/**
 * single body force function.
 */
typedef void (*MxForce_OneBodyPtr)(struct MxForce*, struct MxParticle *, int stateVectorId, FPTYPE*f);

struct Berendsen;
struct Gaussian;
struct Friction;

/**
 * MxForce is a metatype, in that we can have lots of
 * different instances of force functions, that have different attributes, but
 * only have one base type.
 */
struct MxForce {
    MxForce_OneBodyPtr func;

    /**
     * Describes whether this object is a constant force type.
     * @returns true if a symbol, false otherwise.
     */
    virtual bool isConstant() { return false; }

    static Berendsen* berenderson_tstat(const float &tau);
    static Gaussian* random(const float &std, const float &mean, const float &duration=0.01);
    static Friction* friction(const float &coef, const float &std=0.0, const float &mean=0.0, const float &duration=0.1);
};

/**
 * a binding of a force to a particle type, where we use a coupling constant from the
 * state vector as a scaling.
 */
struct MxForceSingleBinding {
    MxForce *force;
    int stateVectorIndex;
};

struct MxConstantForce;
using MxUserForceFuncType = MxVector3f(*)(MxConstantForce*);

/**
 * a force function defined by a user function, we update the force
 * according to update frequency.
 *
 * this object acts like a constant force, but also acts like a time event,
 * in that it periodically calls a user function to update the force.
 */
struct MxConstantForce : MxForce {
    MxUserForceFuncType *userFunc;
    float updateInterval;
    double lastUpdate;
    
    MxVector3f force;
    
    /**
     * notify this user force object of a simulation time step,
     *
     * this will check if interval has elapsed, and update the function.
     *
     * throws std::exception if userfunc is not a valid kind.
     */
    virtual void onTime(double time);

    virtual MxVector3f getValue();
    
    /**
     * sets the value of the force to a vector
     *
     * throws std::exception if invalid value.
     */
    void setValue(const MxVector3f &f);
    
    /**
     * sets the value of the force from a user function. 
     * if a user function is passed, then it is stored as the user function of the force
     *
     * throws std::exception if invalid value.
     */
    void setValue(MxUserForceFuncType *_userFunc=NULL);

    float getPeriod();
    void setPeriod(const float &period);

    bool isConstant() { return true; }

    MxConstantForce();
    MxConstantForce(const MxVector3f &f, const float &period=std::numeric_limits<float>::max());
    MxConstantForce(MxUserForceFuncType *f, const float &period=std::numeric_limits<float>::max());
    virtual ~MxConstantForce(){}
};

struct MxConstantForcePy : MxConstantForce {

    MxConstantForcePy();
    MxConstantForcePy(const MxVector3f &f, const float &period=std::numeric_limits<float>::max());
    MxConstantForcePy(PyObject *f, const float &period=std::numeric_limits<float>::max());
    virtual ~MxConstantForcePy();

    void onTime(double time);
    MxVector3f getValue();

    void setValue(PyObject *_userFunc=NULL);

private:

    PyObject *callable;
};

struct Berendsen : MxForce {
    float itau;
};

struct Gaussian : MxForce {
    float std;
    float mean;
    unsigned durration_steps;
};

struct Friction : MxForce {
    float coef;
    float std;
    float mean;
    unsigned durration_steps;
};

#endif /* SRC_MDCORE_SRC_MXFORCE_H_ */



/**
 * Old Notes, kept, for now, for reference
 *
 * The simulator uses force function to calculate the total force applied to the geometric
 * physical objects (vertices, triangles, etc...). A compiled model (typically read from
 * a compiled shared library) needs to provide 'force functions' to the simulator.
 *
 * The basic simulation flow is:
 *  * Calculate total force applied to each geometric physical object using the specified
 *    MxForceFunction derived objects.
 *  * Use the total force to calculate time evolution (either F \propto ma or F \propto mv).
 *  * update the (possibly velocities) and positions.
 *
 * *Different kinds of force calculations*
 *
 * Molecular dynamics typically define
 *
 * bonds, angles, dihedrals and impropers.
 *
 * The standard bond is essentially a spring that connects two vertices. We know that this
 * spring acts uniformly on each atom in the pair. We can optimize this calculation by performing
 * the calculation only once, but applying it to both vertices at the same time (in opposite
 * directions of course).
 *
 *
 *
 * Volume Energy
 * Acts in the opposite direction of the surface normal of vertex or surface triangle.
 * This is a function of a given surface location and the cell that the surface element
 * belongs to.
 *
 *
 *
 * Surface Energy
 * The net force due to surface tension of a perfectly flat sheet is zero at every interior
 * point. There is no motion of surface elements on sheets at held in a fixed positions,
 * hence the surface tension force sums to zero. On curved surfaces, surface tension acts to
 * pull every surface element towards their neighboring surface elements. On a sphere, force
 * due to surface tension points exactly towards the center of the sphere. Surface tension
 * will tend to make the surface locally flat.
 *
 * Bending Energy
 *
 *
 *
 */
