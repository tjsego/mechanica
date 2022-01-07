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
#ifndef INCLUDE_DIHEDRAL_H_
#define INCLUDE_DIHEDRAL_H_

#include "platform.h"
#include "mdcore_config.h"

#include <random>

/* dihedral error codes */
#define dihedral_err_ok                    0
#define dihedral_err_null                  -1
#define dihedral_err_malloc                -2


/** ID of the last error */
CAPI_DATA(int) dihedral_err;

struct MxDihedralHandle;
struct MxParticleHandle;

/** The dihedral structure */
typedef struct MxDihedral {

	/* ids of particles involved */
	int i, j, k, l;

    uint64_t creation_time;

	/**
	 * half life decay time for this bond.
	 */
	double half_life;

	/* potential energy required to break this bond */
	double dissociation_energy;

	/* potential energy of this bond */
	double potential_energy;

	/* dihedral potential. */
	struct MxPotential *potential;

    struct NOMStyle *style;

	void init(MxPotential *potential, 
              MxParticleHandle *p1, 
              MxParticleHandle *p2, 
              MxParticleHandle *p3, 
              MxParticleHandle *p4);

    /**
     * @brief Creates a dihedral bond
     * 
     * @param potential potential of the bond
     * @param p1 first outer particle
     * @param p2 first center particle
     * @param p3 second center particle
	 * @param p4 second outer particle
     * @return MxDihedralHandle* 
     */
    static MxDihedralHandle *create(MxPotential *potential, 
                                 	MxParticleHandle *p1, 
                                 	MxParticleHandle *p2, 
                                 	MxParticleHandle *p3, 
                                 	MxParticleHandle *p4);

    /**
     * @brief Get a JSON string representation
     * 
     * @return std::string 
     */
    std::string toString();

    /**
     * @brief Create from a JSON string representation. 
     * 
     * The returned dihedral is not automatically registered with the engine. 
     * 
     * @param str 
     * @return MxDihedral* 
     */
    static MxDihedral *fromString(const std::string &str);

} MxDihedral;

struct MxDihedralHandle {
	int id;

    /**
     * @brief Gets the dihedral of this handle
     * 
     * @return MxDihedral* 
     */
    MxDihedral *dihedral();

    std::string str();

    /**
     * @brief Destroy the dihedral
     * 
     * @return HRESULT 
     */
    HRESULT destroy();

    /**
     * @brief Gets all dihedrals in the universe
     * 
     * @return std::vector<MxDihedralHandle*> 
     */
    static std::vector<MxDihedralHandle*> items();

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
    float getDissociationEnergy();
    void setDissociationEnergy(const float &dissociation_energy);
    float getHalfLife();
    void setHalfLife(const float &half_life);
    NOMStyle *getStyle();
    void setStyle(NOMStyle *style);
    double getAge();

    MxDihedralHandle() : id(-1) {}
    MxDihedralHandle(const int &_id) : id(_id) {}
};

/**
 * @brief Shared global dihedral style
 * 
 */
CAPI_DATA(NOMStyle*) MxDihedral_StylePtr;

/**
 * @brief Destroys a dihedral
 * 
 * @param d dihedral to destroy
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxDihedral_Destroy(MxDihedral *d);

/**
 * @brief Destroys all dihedrals in the universe
 * 
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxDihedral_DestroyAll();

/**
 * @brief Tests whether a dihedral decays
 * 
 * @param d dihedral to test
 * @param uniform01 uniform random distribution; optional
 * @return true if the bond decays
 */
bool MxDihedral_decays(MxDihedral *d, std::uniform_real_distribution<double> *uniform01=NULL);

/* associated functions */
int dihedral_eval ( struct MxDihedral *d , int N , struct engine *e , double *epot_out );
int dihedral_evalf ( struct MxDihedral *d , int N , struct engine *e , FPTYPE *f , double *epot_out );


namespace mx { namespace io {

template <>
HRESULT toFile(const MxDihedral &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxDihedral *dataElement);

}};

#endif // INCLUDE_DIHEDRAL_H_
