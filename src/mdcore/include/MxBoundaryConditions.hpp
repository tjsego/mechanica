/*
 * BoundaryConditions.h
 *
 *  Created on: Feb 10, 2021
 *      Author: andy
 */
#pragma once
#ifndef SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_
#define SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_

#include <platform.h>
#include <types/mx_types.h>
#include <io/mx_io.h>

#include <unordered_map>

enum BoundaryConditionKind : unsigned int {
    BOUNDARY_VELOCITY       = 1 << 0,
    BOUNDARY_PERIODIC       = 1 << 1,
    BOUNDARY_FREESLIP       = 1 << 2,
    BOUNDARY_POTENTIAL      = 1 << 3,
    BOUNDARY_NO_SLIP        = 1 << 4, // really just velocity with zero velocity
    BOUNDARY_RESETTING      = 1 << 5, // reset the chemical cargo when particles cross boundaries. 
    BOUNDARY_ACTIVE         = BOUNDARY_FREESLIP | BOUNDARY_VELOCITY | BOUNDARY_POTENTIAL
};

struct MxParticleType;
struct MxPotential;

/**
 * @brief A condition on a boundary of the universe. 
 * 
 */
struct CAPI_EXPORT MxBoundaryCondition {
    BoundaryConditionKind kind;

    // id of this boundary, id's go from 0 to 6 (top, bottom, etc..)
    int id;

    /**
     * @brief the velocity on the boundary
     */
    MxVector3f velocity;

    /** 
     * @brief restoring percent. 
     * 
     * When objects hit this boundary, they get reflected back at `restore` percent, 
     * so if restore is 0.5, and object hitting the boundary at 3 length / time 
     * recoils with a velocity of 1.5 lengths / time. 
     */
    float restore;

    /**
     * @brief name of the boundary
     */
    const char* name;
    
    /**
     * @brief vector normal to the boundary
     */
    MxVector3f normal;

    /**
     * pointer to offset in main array allocated in MxBoundaryConditions.
     */
    struct MxPotential **potenntials;

    // many potentials act on the sum of both particle radii, so this
    // paramter makes it looks like the wall has a sheet of particles of
    // radius.
    float radius;

    /**
     * sets the potential for the given particle type.
     */
    void set_potential(struct MxParticleType *ptype, struct MxPotential *pot);

    std::string kindStr() const;
    std::string str(bool show_name) const;

    unsigned init(const unsigned &kind);
    unsigned init(const MxVector3f &velocity, const float *restore=NULL);
    unsigned init(const std::unordered_map<std::string, unsigned int> vals, 
                  const std::unordered_map<std::string, MxVector3f> vels, 
                  const std::unordered_map<std::string, float> restores);
};

/**
 * @brief The BoundaryConditions class serves as a container for the six 
 * instances of the :class:`MxBoundaryCondition` object
 * 
 */
struct CAPI_EXPORT MxBoundaryConditions {

    /**
     * @brief The top boundary
     */
    MxBoundaryCondition top;

    /**
     * @brief The bottom boundary
     */
    MxBoundaryCondition bottom;

    /**
     * @brief The left boundary
     */
    MxBoundaryCondition left;

    /**
     * @brief The right boundary
     */
    MxBoundaryCondition right;

    /**
     * @brief The front boundary
     */
    MxBoundaryCondition front;

    /**
     * @brief The back boundary
     */
    MxBoundaryCondition back;

    // pointer to big array of potentials, 6 * max types.
    // each boundary condition has a pointer that's an offset
    // into this array, so allocate and free in single block.
    // allocated in MxBoundaryConditions_Init.
    struct MxPotential **potenntials;

    MxBoundaryConditions() {}
    MxBoundaryConditions(int *cells);
    MxBoundaryConditions(int *cells, const int &value);
    MxBoundaryConditions(int *cells, 
                         const std::unordered_map<std::string, unsigned int> vals, 
                         const std::unordered_map<std::string, MxVector3f> vels, 
                         const std::unordered_map<std::string, float> restores);

    /**
     * @brief sets a potential for ALL boundary conditions and the given potential.
     * 
     * @param ptype particle type
     * @param pot potential
     */
    void set_potential(struct MxParticleType *ptype, struct MxPotential *pot);

    /**
     * bitmask of periodic boundary conditions
     */
    uint32_t periodic;

    static unsigned boundaryKindFromString(const std::string &s);
    static unsigned boundaryKindFromStrings(const std::vector<std::string> &kinds);

    std::string str();

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * @param str 
     * @return MxBoundaryConditions* 
     */
    static MxBoundaryConditions *fromString(const std::string &str);

private:

    HRESULT _initIni();
    HRESULT _initFin(int *cells);

    // processes directional initialization inputs
    void _initDirections(const std::unordered_map<std::string, unsigned int> vals);

    // processes sides initialization inputs
    void _initSides(const std::unordered_map<std::string, unsigned int> vals, 
                    const std::unordered_map<std::string, MxVector3f> vels, 
                    const std::unordered_map<std::string, float> restores);
};

struct CAPI_EXPORT MxBoundaryConditionsArgsContainer {
    int *bcValue;
    std::unordered_map<std::string, unsigned int> *bcVals;
    std::unordered_map<std::string, MxVector3f> *bcVels;
    std::unordered_map<std::string, float> *bcRestores;

    void setValueAll(const int &_bcValue);
    void setValue(const std::string &name, const unsigned int &value);
    void setVelocity(const std::string &name, const MxVector3f &velocity);
    void setRestore(const std::string &name, const float restore);

    MxBoundaryConditions *create(int *cells);

    MxBoundaryConditionsArgsContainer(int *_bcValue=NULL, 
                                      std::unordered_map<std::string, unsigned int> *_bcVals=NULL, 
                                      std::unordered_map<std::string, MxVector3f> *_bcVels=NULL, 
                                      std::unordered_map<std::string, float> *_bcRestores=NULL);

private:
    void switchType(const bool &allSides);
};

/**
 * a particle moved from one cell to another, this checks if its a periodic
 * crossing, and adjusts any particle state values if the boundaries say so.
 */
void apply_boundary_particle_crossing(struct MxParticle *p, const int *delta,
                                     const struct space_cell *source_cell, const struct space_cell *dest_cell);


namespace mx { namespace io {

template <>
HRESULT toFile(const MxBoundaryCondition &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryCondition *dataElement);

template <>
HRESULT toFile(const MxBoundaryConditions &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

// Requires returned value to already be initialized with cells
template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryConditions *dataElement);

// Takes a file element generated from MxBoundaryConditions
template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBoundaryConditionsArgsContainer *dataElement);

}};

#endif /* SRC_MDCORE_SRC_BOUNDARYCONDITIONS_H_ */
