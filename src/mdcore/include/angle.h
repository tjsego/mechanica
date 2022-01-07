/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/
#ifndef INCLUDE_ANGLE_H
#define INCLUDE_ANGLE_H
#include "platform.h"

/* Include configuration header */
#include "mdcore_config.h"
#include <MxPotential.h>

#include <random>
#include <string>

/* angle error codes */
#define angle_err_ok                     0
#define angle_err_null                  -1
#define angle_err_malloc                -2

/** ID of the last error */
CAPI_DATA(int) angle_err;


typedef enum MxAngleFlags {

    // none type angles are initial state, and can be
    // re-assigned if ref count is 1 (only owned by engine).
    ANGLE_NONE                   = 0,
    ANGLE_ACTIVE                 = 1 << 0,
    ANGLE_FOO   = 1 << 1,
} MxAngleFlags;

struct MxAngleHandle;
struct MxParticleHandle;

/**
 * @brief A bond concerning an angle
 * 
 * If you're building a model, you should probably instead be working with a 
 * MxAngleHandle. 
 */
typedef struct MxAngle {

    uint32_t flags;

	/* ids of particles involved */
	int i, j, k;

    uint64_t creation_time;

	/**
	 * half life decay time for this bond.
	 */
	double half_life;

	/* potential energy required to break this bond */
	double dissociation_energy;

	/* potential energy of this bond */
	double potential_energy;

	/* id of the potential. */
	struct MxPotential *potential;

    struct NOMStyle *style;

    void init(MxPotential *potential, 
              MxParticleHandle *p1, 
              MxParticleHandle *p2, 
              MxParticleHandle *p3, 
              uint32_t flags=0);

    /**
     * @brief Creates an angle bond. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param potential potential of the bond
     * @param p1 first outer particle
     * @param p2 center particle
     * @param p3 second outer particle
     * @param flags angle flags
     * @return MxAngleHandle* 
     */
    static MxAngleHandle *create(MxPotential *potential, 
                                 MxParticleHandle *p1, 
                                 MxParticleHandle *p2, 
                                 MxParticleHandle *p3, 
                                 uint32_t flags=0);

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * The returned angle is not automatically registered with the engine. 
     * 
     * @param str 
     * @return MxAngle* 
     */
    static MxAngle *fromString(const std::string &str);

} MxAngle;

/**
 * @brief A handle to an angle bond
 * 
 * This is a safe way to work with an angle bond. 
 */
struct MxAngleHandle {
    int id;

    /**
     * @brief Gets the angle of this handle
     * 
     * @return MxAngle* 
     */
    MxAngle *angle();

    std::string str();

    /**
     * @brief Destroy the angle. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @return HRESULT 
     */
    HRESULT destroy();

    /**
     * @brief Gets all angles in the universe
     * 
     * @return std::vector<MxAngleHandle*> 
     */
    static std::vector<MxAngleHandle*> items();

    /**
     * @brief Tests whether this bond decays
     * 
     * @return true when the bond should decay
     */
    bool decays();

    MxParticleHandle *operator[](unsigned int index);

    double getEnergy();
    std::vector<int32_t> getParts();
    MxPotential *getPotential();
    uint32_t getId();
    float getDissociationEnergy();
    void setDissociationEnergy(const float &dissociation_energy);
    float getHalfLife();
    void setHalfLife(const float &half_life);
    bool getActive();
    NOMStyle *getStyle();
    void setStyle(NOMStyle *style);
    double getAge();

    MxAngleHandle() : id(-1) {}
    MxAngleHandle(const int &_id) : id(_id) {}
};

// todo: implement new angle interactions when supported by engine (engine_angle_add)
#if 0
/**
 * @brief Add a angle interaction to the engine.
 *
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pid Index of the #potential for this bond.
 *
 * Note, the potential (pid) has to be previously added by engine_angle_addpot.
 */
CAPI_FUNC(MxAngle*) MxAngle_NewFromIds(int i , int j , int k , int pid );

/**
 * @brief Add a angle interaction to the engine.
 *
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pot An existing potential.
 *
 * This checks if the potential is already in the engine, and if so, uses it,
 * otherwise, adds the potential to the engine.
 */
CAPI_FUNC(MxAngle*) MxAngle_NewFromIdsAndPotential(int i , int j , int k , struct MxPotential *pot);
#endif

/**
 * @brief Shared global angle style
 * 
 */
CAPI_DATA(NOMStyle*) MxAngle_StylePtr;

/**
 * @brief Destroys an angle
 * 
 * @param a angle to destroy
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxAngle_Destroy(MxAngle *a);

/**
 * @brief Destroys all angles in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxAngle_DestroyAll();

/**
 * @brief Tests whether an angle decays
 * 
 * @param a angle to test
 * @param uniform01 uniform random distribution; optional
 * @return true if the bond decays
 */
CAPI_FUNC(bool) MxAngle_decays(MxAngle *a, std::uniform_real_distribution<double> *uniform01=NULL);

/* associated functions */
int angle_eval ( struct MxAngle *a , int N , struct engine *e , double *epot_out );
int angle_evalf ( struct MxAngle *a , int N , struct engine *e , FPTYPE *f , double *epot_out );


namespace mx { namespace io {

template <>
HRESULT toFile(const MxAngle &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxAngle *dataElement);

}};

#endif // INCLUDE_ANGLE_H
