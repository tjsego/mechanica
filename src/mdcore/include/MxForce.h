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
 * @brief MxForce is a metatype, in that Mechanica has lots of 
 * different instances of force functions, that have different attributes, but
 * only have one base type. 
 * 
 * Forces are one of the fundamental processes in Mechanica that cause objects to move. 
 */
struct MxForce {
    MxForce_OneBodyPtr func;

    /**
     * @brief Tests whether this object is a constant force type.
     * 
     * @return true if constant
     */
    virtual bool isConstant() { return false; }

    /**
     * @brief Creates a Berendsen thermostat. 
     * 
     * The thermostat uses the target temperature @f$ T_0 @f$ from the object 
     * to which it is bound. 
     * The Berendsen thermostat effectively re-scales the velocities of an object in 
     * order to make the temperature of that family of objects match a specified 
     * temperature.
     * 
     * The Berendsen thermostat force has the function form: 
     * 
     * @f[
     * 
     *      \frac{\mathbf{p}}{\tau_T} \left(\frac{T_0}{T} - 1 \right),
     * 
     * @f]
     * 
     * where @f$ \mathbf{p} @f$ is the momentum, 
     * @f$ T @f$ is the measured temperature of a family of 
     * particles, @f$ T_0 @f$ is the control temperature, and 
     * @f$ \tau_T @f$ is the coupling constant. The coupling constant is a measure 
     * of the time scale on which the thermostat operates, and has units of 
     * time. Smaller values of @f$ \tau_T @f$ result in a faster acting thermostat, 
     * and larger values result in a slower acting thermostat.
     * 
     * @param tau time constant that determines how rapidly the thermostat effects the system.
     * @return Berendsen* 
     */
    static Berendsen* berenderson_tstat(const float &tau);

    /**
     * @brief Creates a random force. 
     * 
     * A random force has a randomly selected orientation and magnitude. 
     * 
     * Orientation is selected according to a uniform distribution on the unit sphere. 
     * 
     * Magnitude is selected according to a prescribed mean and standard deviation. 
     * 
     * @param std standard deviation of magnitude
     * @param mean mean of magnitude
     * @param duration duration of force. Defaults to 0.01. 
     * @return Gaussian* 
     */
    static Gaussian* random(const float &std, const float &mean, const float &duration=0.01);

    /**
     * @brief Creates a friction force. 
     * 
     * A friction force has the form: 
     * 
     * @f[
     * 
     *      - \frac{|| \mathbf{v} ||}{\tau} \mathbf{v} + \mathbf{f}_{r} ,
     * 
     * @f]
     * 
     * where @f$ \mathbf{v} @f$ is the velocity of a particle, @f$ \tau @f$ is a time constant and 
     * @f$ \mathbf{f}_r @f$ is a random force. 
     * 
     * @param coef time constant
     * @param std standard deviation of random force magnitude
     * @param mean mean of random force magnitude
     * @param duration duration of force. Defaults to 0.1. 
     * @return Friction* 
     */
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
 * @brief A custom force function. 
 * 
 * The force is updated according to an update frequency.
 * 
 * This object acts like a constant force, but also acts like a time event,
 * in that it periodically calls a custom function to update the applied force. 
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

    /**
     * @brief Creates an instance from an underlying custom python function
     * 
     * @param f python function. Takes no arguments and returns a three-component vector. 
     * @param period period at which the force is updated. 
     */
    MxConstantForcePy(PyObject *f, const float &period=std::numeric_limits<float>::max());
    virtual ~MxConstantForcePy();

    void onTime(double time);
    MxVector3f getValue();

    void setValue(PyObject *_userFunc=NULL);

private:

    PyObject *callable;
};

/**
 * @brief Berendsen force. 
 * 
 * Create one with :meth:`MxForce.berenderson_tstat`. 
 */
struct Berendsen : MxForce {
    /**
     * @brief time constant
     */
    float itau;
};

/**
 * @brief Random force. 
 * 
 * Create one with :meth:`MxForce.random`. 
 */
struct Gaussian : MxForce {
    /**
     * @brief standard deviation of magnitude
     */
    float std;

    /**
     * @brief mean of magnitude
     */
    float mean;

    /**
     * @brief duration of force.
     */
    unsigned durration_steps;
};

/**
 * @brief Friction force. 
 * 
 * Create one with :meth:`MxForce.friction`. 
 */
struct Friction : MxForce {
    /**
     * @brief time constant
     */
    float coef;

    /**
     * @brief standard deviation of random force magnitude
     */
    float std;

    /**
     * @brief mean of random force magnitude
     */
    float mean;

    /**
     * @brief duration of force, in time steps
     */
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
