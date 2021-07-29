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

/** The bond structure */
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

	struct MxPotential *potential;
    
    struct NOMStyle *style;

    /**
     * @brief Construct a new bond handle and underlying bond
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

} MxBond;

struct MxParticleType;
struct MxParticleList;

struct MxBondHandle {
    int32_t id;
    
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
     * @brief Construct a new bond handle and underlying bond
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
     * @brief For initializing a bond after constructing with default constructor
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
    std::string str();
    bool check();
    
    static std::vector<MxBondHandle*>* pairwise(MxPotential* pot,
                                                MxParticleList *parts,
                                                const double &cutoff,
                                                std::vector<std::pair<MxParticleType*, MxParticleType*>* > *ppairs,
                                                const double &half_life,
                                                const double &bond_energy,
                                                uint32_t flags);
    HRESULT destroy();
    static std::vector<MxBondHandle*> bonds();
    static std::vector<MxBondHandle*> items();

    MxParticleHandle *operator[](unsigned int index);
    
    double getEnergy();
    std::vector<int32_t> getParts();
    MxPotential *getPotential();
    uint32_t getId();
    float getDissociationEnergy();
    bool getActive();
    NOMStyle *getStyle();

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
CAPI_DATA(NOMStyle*) MxBond_StylePtr;

/**
 * deletes, marks a bond ready for deleteion, removes the potential,
 * other vars, clears the bond, and makes is ready to be
 * over-written.
 */
CAPI_FUNC(HRESULT) MxBond_Destroy(MxBond *b);

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

#endif // INCLUDE_BOND_H_
