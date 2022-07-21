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
#include <types/mx_types.h>
#include <io/mx_io.h>

#include <limits>

enum MXFORCE_KIND {
    MXFORCE_ONEBODY,
    MXFORCE_PAIRWISE
};

enum MXFORCE_TYPE {
    FORCE_FORCE         = 0, 
    FORCE_BERENDSEN     = 1 << 0, 
    FORCE_GAUSSIAN      = 1 << 1, 
    FORCE_FRICTION      = 1 << 2, 
    FORCE_SUM           = 1 << 3, 
    FORCE_CONSTANT      = 1 << 4
};

/**
 * single body force function.
 */
typedef void (*MxForce_EvalFcn)(struct MxForce*, struct MxParticle*, FPTYPE*);

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
struct CAPI_EXPORT MxForce {
    MXFORCE_TYPE type = FORCE_FORCE;

    MxForce_EvalFcn func;

    int stateVectorIndex = -1;

    /**
     * @brief Tests whether this object is a constant force type.
     * 
     * @return true if constant
     */
    virtual bool isConstant() { return false; }

    /**
     * @brief Bind a force to a species. 
     * 
     * When a force is bound to a species, the magnitude of the force is scaled by the concentration of the species. 
     * 
     * @param a_type particle type containing the species
     * @param coupling_symbol symbol of the species
     * @return HRESULT 
     */
    HRESULT bind_species(struct MxParticleType *a_type, const std::string &coupling_symbol);

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
    static Berendsen* berendsen_tstat(const float &tau);

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
     *      - \frac{|| \mathbf{v} ||}{\tau} \mathbf{v} ,
     * 
     * @f]
     * 
     * where @f$ \mathbf{v} @f$ is the velocity of a particle and @f$ \tau @f$ is a time constant. 
     * 
     * @param coef time constant
     * @return Friction* 
     */
    static Friction* friction(const float &coef);

    MxForce& operator+(const MxForce& rhs);

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    virtual std::string toString();

    /**
     * @brief Create from a JSON string representation
     * 
     * @param str 
     * @return MxForce* 
     */
    static MxForce *fromString(const std::string &str);
};

struct CAPI_EXPORT MxForceSum : MxForce {
    MxForce *f1, *f2;

    /**
     * @brief Convert basic force to force sum. 
     * 
     * If the basic force is not a force sum, then NULL is returned. 
     * 
     * @param f 
     * @return MxForceSum* 
     */
    static MxForceSum *fromForce(MxForce *f);
};

CAPI_FUNC(MxForce*) MxForce_add(MxForce *f1, MxForce *f2);

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
struct CAPI_EXPORT MxConstantForce : MxForce {
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

    /**
     * @brief Convert basic force to MxConstantForce. 
     * 
     * If the basic force is not a MxConstantForce, then NULL is returned. 
     * 
     * @param f 
     * @return MxConstantForce* 
     */
    static MxConstantForce *fromForce(MxForce *f);
};

/**
 * @brief Berendsen force. 
 * 
 * Create one with :meth:`MxForce.berendsen_tstat`. 
 */
struct CAPI_EXPORT Berendsen : MxForce {
    /**
     * @brief time constant
     */
    float itau;

    /**
     * @brief Convert basic force to Berendsen. 
     * 
     * If the basic force is not a Berendsen, then NULL is returned. 
     * 
     * @param f 
     * @return Berendsen* 
     */
    static Berendsen *fromForce(MxForce *f);
};

/**
 * @brief Random force. 
 * 
 * Create one with :meth:`MxForce.random`. 
 */
struct CAPI_EXPORT Gaussian : MxForce {
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

    /**
     * @brief Convert basic force to Gaussian. 
     * 
     * If the basic force is not a Gaussian, then NULL is returned. 
     * 
     * @param f 
     * @return Gaussian* 
     */
    static Gaussian *fromForce(MxForce *f);
};

/**
 * @brief Friction force. 
 * 
 * Create one with :meth:`MxForce.friction`. 
 */
struct CAPI_EXPORT Friction : MxForce {
    /**
     * @brief time constant
     */
    float coef;

    /**
     * @brief Convert basic force to Friction. 
     * 
     * If the basic force is not a Friction, then NULL is returned. 
     * 
     * @param f 
     * @return Friction* 
     */
    static Friction *fromForce(MxForce *f);
};


namespace mx { namespace io {

template <>
HRESULT toFile(const MXFORCE_TYPE &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MXFORCE_TYPE *dataElement);

template <>
HRESULT toFile(const MxConstantForce &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxConstantForce *dataElement);

template <>
HRESULT toFile(const MxForceSum &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxForceSum *dataElement);

template <>
HRESULT toFile(const Berendsen &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Berendsen *dataElement);

template <>
HRESULT toFile(const Gaussian &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Gaussian *dataElement);

template <>
HRESULT toFile(const Friction &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, Friction *dataElement);

template <>
HRESULT toFile(MxForce *dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxForce **dataElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, std::vector<MxForce*> *dataElement);

}};


MxForceSum *MxForceSum_fromStr(const std::string &str);
Berendsen *Berendsen_fromStr(const std::string &str);
Gaussian *Gaussian_fromStr(const std::string &str);
Friction *Friction_fromStr(const std::string &str);


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
