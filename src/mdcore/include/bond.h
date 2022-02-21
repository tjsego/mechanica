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

#ifndef INCLUDE_BOND_H_
#define INCLUDE_BOND_H_
#include "platform.h"
#include <random>

/* bond error codes */
#define bond_err_ok                    0
#define bond_err_null                  -1
#define bond_err_malloc                -2


/** ID of the last error */
CAPI_DATA(int) bond_err;


typedef enum MxBondFlags {

    // none type bonds are initial state, and can be
    // re-assigned if ref count is 1 (only owned by engine).
    BOND_NONE                   = 0,
    // a non-active and will be over-written in the
    // next bond_alloc call.
    BOND_ACTIVE                 = 1 << 0,
} MxBondFlags;

// list of pairs...
struct Pair {
    int32_t i;
    int32_t j;
};

typedef std::vector<Pair> PairList;
struct MxBondHandle;
struct MxParticleHandle;

/**
 * @brief Bonds apply a potential to a particular set of particles. 
 * 
 * If you're building a model, you should probably instead be working with a 
 * MxBondHandle. 
 */
typedef struct MxBond {

    uint32_t flags;

	/* ids of particles involved */
	int32_t i, j;
    
    uint32_t id;

    uint64_t creation_time;

	/**
	 * half life decay time for this bond.
	 */
	double half_life;

	/* potential energy required to break this bond */
	double dissociation_energy;

	/* potential energy of this bond */
	double potential_energy;

	struct MxPotential *potential;
    
    struct MxStyle *style;

    /**
     * @brief Construct a new bond handle and underlying bond. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param potential bond potential
     * @param i ith particle
     * @param j jth particle
     * @param half_life bond half life
     * @param dissociation_energy dissociation energy
     * @param flags bond flags
     */
    static MxBondHandle *create(struct MxPotential *potential, 
                                MxParticleHandle *i, 
                                MxParticleHandle *j, 
                                double *half_life=NULL, 
                                double *dissociation_energy=NULL, 
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
     * The returned bond is not automatically registered with the engine. 
     * 
     * @param str 
     * @return MxBond* 
     */
    static MxBond *fromString(const std::string &str);

} MxBond;

struct MxParticleType;
struct MxParticleList;

/**
 * @brief Handle to a bond
 * 
 * This is a safe way to work with a bond. 
 */
struct MxBondHandle {
    int32_t id;
    
    /**
     * @brief Gets the underlying bond
     * 
     * @return MxBond* 
     */
    MxBond *get();

    /**
     * @brief Construct a new bond handle and do nothing
     * Subsequent usage will require a call to 'init'
     * 
     */
    MxBondHandle() : id(-1) {};

    /**
     * @brief Construct a new bond handle from an existing bond id
     * 
     * @param id id of existing bond
     */
    MxBondHandle(int id);

    /**
     * @brief Construct a new bond handle and underlying bond. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param potential bond potential
     * @param i id of ith particle
     * @param j id of jth particle
     * @param half_life bond half life
     * @param bond_energy bond energy
     * @param flags bond flags
     */
    MxBondHandle(struct MxPotential *potential, 
                 int32_t i, 
                 int32_t j, 
                 double half_life, 
                 double bond_energy, 
                 uint32_t flags);

    /**
     * @brief For initializing a bond after constructing with default constructor. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param pot bond potential
     * @param p1 ith particle
     * @param p2 jth particle
     * @param half_life bond half life
     * @param bond_energy bond energy
     * @param flags bond flags
     * @return int 
     */
    int init(MxPotential *pot, 
             MxParticleHandle *p1, 
             MxParticleHandle *p2, 
             const double &half_life=std::numeric_limits<double>::max(), 
             const double &bond_energy=std::numeric_limits<double>::max(), 
             uint32_t flags=0);
    
    /**
     * @brief Get a summary string of the bond
     * 
     * @return std::string 
     */
    std::string str();

    /**
     * @brief Check the validity of the handle
     * 
     * @return true if ok
     * @return false 
     */
    bool check();
    
    /**
     * @brief Apply bonds to a list of particles. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @param pot the potential of the created bonds
     * @param parts list of particles
     * @param cutoff cutoff distance of particles that are bonded
     * @param ppairs type pairs of bonds
     * @param half_life bond half life
     * @param bond_energy bond energy
     * @param flags bond flags
     * @return std::vector<MxBondHandle*>* 
     */
    static std::vector<MxBondHandle*>* pairwise(MxPotential* pot,
                                                MxParticleList *parts,
                                                const double &cutoff,
                                                std::vector<std::pair<MxParticleType*, MxParticleType*>* > *ppairs,
                                                const double &half_life,
                                                const double &bond_energy,
                                                uint32_t flags);
    
    /**
     * @brief Destroy the bond. 
     * 
     * Automatically updates when running on a CUDA device. 
     * 
     * @return HRESULT 
     */
    HRESULT destroy();
    static std::vector<MxBondHandle*> bonds();

    /**
     * @brief Gets all bonds in the universe
     * 
     * @return std::vector<MxBondHandle*> 
     */
    static std::vector<MxBondHandle*> items();

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
    MxStyle *getStyle();
    void setStyle(MxStyle *style);
    double getAge();

private:

    int _init(uint32_t flags, 
              int32_t i, 
              int32_t j, 
              double half_life, 
              double bond_energy, 
              struct MxPotential *potential);
};

bool contains_bond(const std::vector<MxBondHandle*> &bonds, int a, int b);

/**
 * shared global bond style
 */
CAPI_DATA(MxStyle*) MxBond_StylePtr;

/**
 * deletes, marks a bond ready for deleteion, removes the potential,
 * other vars, clears the bond, and makes is ready to be
 * over-written. 
 * 
 * Automatically updates when running on a CUDA device. 
 */
CAPI_FUNC(HRESULT) MxBond_Destroy(MxBond *b);

/**
 * @brief Deletes all bonds in the universe. 
 * 
 * Automatically updates when running on a CUDA device. 
 * 
 * @return HRESULT 
 */
CAPI_FUNC(HRESULT) MxBond_DestroyAll();

/**
 * @brief Tests whether a bond decays
 * 
 * @param b bond to test
 * @param uniform01 uniform random distribution; optional
 * @return true if the bond decays
 */
CAPI_FUNC(bool) MxBond_decays(MxBond *b, std::uniform_real_distribution<double> *uniform01=NULL);

HRESULT MxBond_Energy (MxBond *b, double *epot_out);

/* associated functions */
CAPI_FUNC(int) bond_eval (MxBond *b , int N , struct engine *e , double *epot_out );
CAPI_FUNC(int) bond_evalf (MxBond *b , int N , struct engine *e , FPTYPE *f , double *epot_out );


/**
 * find all the bonds that interact with the given particle id
 */
std::vector<int32_t> MxBond_IdsForParticle(int32_t pid);

int insert_bond(std::vector<MxBondHandle*> &bonds, int a, int b,
                MxPotential *pot, MxParticleList *parts);


namespace mx { namespace io {

template <>
HRESULT toFile(const MxBond &dataElement, const MxMetaData &metaData, MxIOElement *fileElement);

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxBond *dataElement);

}};

#endif // INCLUDE_BOND_H_
