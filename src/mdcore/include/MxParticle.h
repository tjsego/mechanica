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
#ifndef INCLUDE_PARTICLE_H_
#define INCLUDE_PARTICLE_H_
#include "platform.h"
#include "fptype.h"
#include "../../state/MxStateVector.h"
#include "../../types/mx_types.h"
#include "space_cell.h"
#include "bond.h"
#include "MxParticleList.hpp"
#include <set>


CAPI_STRUCT(NOMStyle);


/* error codes */
#define PARTICLE_ERR_OK                 0
#define PARTICLE_ERR_NULL              -1
#define PARTICLE_ERR_MALLOC            -2

typedef enum MxParticleTypeFlags {
    PARTICLE_TYPE_NONE          = 0,
    PARTICLE_TYPE_INERTIAL      = 1 << 0,
    PARTICLE_TYPE_DISSAPATIVE   = 1 << 1,
} MxParticleTypeFlags;

typedef enum MxParticleDynamics {
    PARTICLE_NEWTONIAN          = 0,
    PARTICLE_OVERDAMPED         = 1,
} MxParticleDynamics;

/* particle flags */
typedef enum MxParticleFlags {
    PARTICLE_NONE          = 0,
    PARTICLE_GHOST         = 1 << 0,
    PARTICLE_CLUSTER       = 1 << 1,
    PARTICLE_BOUND         = 1 << 2,
    PARTICLE_FROZEN_X      = 1 << 3,
    PARTICLE_FROZEN_Y      = 1 << 4,
    PARTICLE_FROZEN_Z      = 1 << 5,
    PARTICLE_FROZEN        = PARTICLE_FROZEN_X | PARTICLE_FROZEN_Y | PARTICLE_FROZEN_Z,
    PARTICLE_LARGE         = 1 << 6,
} MxParticleFlags;



/** ID of the last error. */
// CAPI_DATA(int) particle_err;


/**
 * increment size of cluster particle list.
 */
#define CLUSTER_PARTLIST_INCR 50

struct MxCluster;
struct MxClusterParticleHandle;

/**
 * The particle data structure.
 *
 * Instance vars for each particle.
 *
 * Note that the arrays for @c x, @c v and @c f are 4 entries long for
 * proper alignment.
 *
 * All particles are stored in a series of contiguous blocks of memory that are owned
 * by the space cells. Each space cell has a array of particle structs.
 */
struct MxParticle  {
    
    /**
     * Particle force
     *
     * ONLY the coherent part of the force should go here. We use multi-step
     * integrators, that need to separate the random and coherent forces.
     *
     * Force gets cleared each step, along with number density, so can clear the
     * whole 4 vector here. 
     */
    union {
        FPTYPE f[4] __attribute__ ((aligned (16)));
        MxVector3f force __attribute__ ((aligned (16)));
        
        struct {
            float __dummy0[3];
            float number_density;
        };
    };


	/** Particle velocity */
    union {
        FPTYPE v[4] __attribute__ ((aligned (16)));
        MxVector3f velocity __attribute__ ((aligned (16)));
        
        struct {
            float __dummy1[3];
            float inv_number_density;
        };
    };

    
    /** Particle position */
    union {
        FPTYPE x[4] __attribute__ ((aligned (16)));
        MxVector3f position __attribute__ ((aligned (16)));

        struct {
            float __dummy2[3];
            uint32_t creation_time;
        };
    };
    

    /** random force force */
    union {
        MxVector3f persistent_force __attribute__ ((aligned (16)));
    };

    // inverse mass
    double imass;

    float radius;

    double mass;

	/** individual particle charge, if needed. */
	float q;

	// runge-kutta k intermediates.
	MxVector3f p0;
	MxVector3f v0;
	MxVector3f xk[4];
	MxVector3f vk[4];

	/**
	 * Particle id, virtual id
	 * TODO: not sure what virtual id is...
	 */
	int id, vid;

	/** particle type. */
	int16_t typeId;

	/**
	 * Id of this parts
	 */
	int32_t clusterId;

	/** Particle flags */
	uint16_t flags;

    /**
     * pointer to the python 'wrapper'. Need this because the particle data
     * gets moved around between cells, and python can't hold onto that directly,
     * so keep a pointer to the python object, and update that pointer
     * when this object gets moved.
     *
     * initialzied to null, and only set when .
     */
    struct MxParticleHandle *_pyparticle;

    /**
     * public way of getting the pyparticle. Creates and caches one if
     * it's not there.
     */
    struct MxParticleHandle *py_particle();

    /**
     * list of particle ids that belong to this particle, if it is a cluster.
     */
    int32_t *parts;
    uint16_t nr_parts;
    uint16_t size_parts;

    /**
     * add a particle (id) to this type
     */
    HRESULT addpart(int32_t uid);

    /**
     * removes a particle from this cluster. Sets the particle cluster id
     * to -1, and removes if from this cluster's list.
     */
    HRESULT removepart(int32_t uid);

    inline MxParticle *particle(int i);

    // style pointer, set at object construction time.
    // may be re-set by users later.
    // the base particle type has a default style.
    NOMStyle *style;

    /**
     * pointer to state vector (optional)
     */
    // todo: update implementation through replacing CStateVector with MxStateVector
    struct MxStateVector *state_vector;

    inline MxVector3f global_position();

    inline void set_global_position(const MxVector3f& pos);
    
    /**
     * performs a self-verify, in debug mode raises assertion if not valid
     */
    bool verify();

    /**
     * Limits casting to cluster by type
     */
    operator MxCluster*();
    
    MxParticle();
};

/**
 * iterates over all parts, does a verify
 */
HRESULT MxParticle_Verify();

#ifndef NDEBUG
#define VERIFY_PARTICLES() MxParticle_Verify()
#else
#define VERIFY_PARTICLES()
#endif


/**
 * Layout of the actual Python particle object.
 *
 * The engine allocates particle memory in blocks, and particle
 * values get moved around all the time, so their addresses change.
 *
 * The partlist is always ordered  by id, i.e. partlist[id]  always
 * points to the same particle, even though that particle may move
 * from cell to cell.
 */
struct CAPI_EXPORT MxParticleHandle {
    int id;
    int typeId;
    inline MxParticle *part();
    inline MxParticleType *type();

    MxParticleHandle() : id(0), typeId(0) {}
    MxParticleHandle(const int &id, const int &typeId) : id(id), typeId(typeId) {}

    virtual inline MxParticleHandle* fission();
    virtual inline MxParticleHandle* split();
    inline HRESULT destroy();
    MxVector3f sphericalPosition(MxParticle *particle=NULL, MxVector3f *origin=NULL);
    virtual MxMatrix3f virial(float *radius=NULL);
    inline HRESULT become(MxParticleType *type);
    MxParticleList *neighbors(const float *distance=NULL, const std::vector<MxParticleType> *types=NULL);
    MxParticleList *getBondedNeighbors();
    float distance(MxParticleHandle *_other);
    std::vector<MxBondHandle*> *getBonds();

    inline double getCharge();
    inline void setCharge(const double &charge);
    inline double getMass();
    inline void setMass(const double &mass);
    inline bool getFrozen();
    inline void setFrozen(const bool frozen);
    inline bool getFrozenX();
    inline void setFrozenX(const bool frozen);
    inline bool getFrozenY();
    inline void setFrozenY(const bool frozen);
    inline bool getFrozenZ();
    inline void setFrozenZ(const bool frozen);
    inline NOMStyle *getStyle();
    inline void setStyle(NOMStyle *style);
    inline double getAge();
    inline double getRadius();
    inline void setRadius(const double &radius);
    inline std::string getName();
    inline std::string getName2();
    inline MxVector3f getPosition();
    inline void setPosition(MxVector3f position);
    inline MxVector3f getVelocity();
    inline void setVelocity(MxVector3f velocity);
    inline MxVector3f getForce();
    inline void setForce(MxVector3f force);
    inline int getId();
    inline int16_t getTypeId();
    inline uint16_t getFlags();
    inline MxStateVector *getSpecies();

    /**
     * Limits casting to cluster by type
     */
    operator MxClusterParticleHandle*();
};

/**
 * Structure containing information on each particle species.
 *
 * This is only a definition for the particle *type*, not the actual
 * instance vars like pos, vel, which are stored in part.
 */
struct CAPI_EXPORT MxParticleType {

    static const int MAX_NAME = 64;

    /** ID of this type */
    int16_t id;

    /**
     *  type flags
     */
    uint32_t type_flags;

    /**
     * particle flags, the type initializer sets these, and
     * all new particle instances get a copy of these.
     */
    uint16_t particle_flags;

    /** Constant physical characteristics */
    double mass, imass, charge;

    /** default radius for particles that don't define their own radius */
    double radius;

    /**
     * energy and potential energy of this type, this is updated by the engine
     * each time step.
     */
    double kinetic_energy;

    double potential_energy;

    double target_energy;

    /**
     * minimum radius, if a fission event occurs, it will not spit a particle
     * such that it's radius gets less than this value.
     *
     * defaults to radius
     */
    double minimum_radius;

    /** Nonbonded interaction parameters. */
    double eps, rmin;

    /**
     * what kind of propator does this particle type use?
     */
    unsigned char dynamics;

    /** Name of this particle type. */
    char name[MAX_NAME], name2[MAX_NAME];

    /**
     * list of particles that belong to this type.
     */
    MxParticleList parts;

    /**
     * list of particle types that belong to this type.
     */
    MxParticleTypeList types;

    // style pointer, optional.
    NOMStyle *style;

    /**
     * optional pointer to species list. This is the metadata for the species, define it in the type.
     */
    struct MxSpeciesList *species = 0;

    /**
     * add a particle (id) to this type
     */
    HRESULT addpart(int32_t id);


    /**
     * remove a particle id from this type
     */
    HRESULT del_part(int32_t id);

    /**
     * get the i'th particle that's a member of this type.
     */
    inline MxParticle *particle(int i);

    /**
     * @brief Get all current particle type ids, excluding clusters
     * 
     * @return std::set<int> 
     */
    static std::set<short int> particleTypeIds();

    bool isCluster();

    // Particle constructor
    MxParticleHandle *operator()(MxVector3f *position=NULL,
                                 MxVector3f *velocity=NULL,
                                 int *clusterId=NULL);

    // Particle type constructor; new type is constructed from the definition of the calling type
    MxParticleType* newType(const char *_name);

    // Registers a type with the engine. 
    // Note that this occurs automatically, unless noReg==true in constructor
    virtual HRESULT registerType();

    // A callback for when a type is registered
    virtual void on_register() {}

    // Whether this type is registered
    bool isRegistered();

    MxParticleType(const bool &noReg=false);
    virtual ~MxParticleType() {}

    inline bool getFrozen();
    inline void setFrozen(const bool &frozen);
    inline bool getFrozenX();
    inline void setFrozenX(const bool &frozen);
    inline bool getFrozenY();
    inline void setFrozenY(const bool &frozen);
    inline bool getFrozenZ();
    inline void setFrozenZ(const bool &frozen);
    // temperature is an ensemble property
    inline double getTemperature();
    inline double getTargetTemperature();

    inline MxParticleList *items();
};

CAPI_FUNC(MxParticleType*) MxParticle_GetType();

CAPI_FUNC(MxParticleType*) MxCluster_GetType();

/**
 * Creates a new MxParticleType for the given particle data pointer.
 *
 * This creates a matching python type for an existing particle data,
 * and is usually called when new types are created from C.
 */
MxParticleType *MxParticleType_ForEngine(struct engine *e, double mass , double charge,
                                         const char *name , const char *name2);

/**
 * Creates and initializes a new particle type, adds it to the
 * global engine
 *
 * creates both a new type, and a new data entry in the engine.
 */
MxParticleType* MxParticleType_New(const char *_name);

/**
 * @brief Get a registered particle type by type name
 * 
 * @param name name of particle type
 * @return MxParticleType* 
 */
CAPI_FUNC(MxParticleType*) MxParticleType_FindFromName(const char* name);

/**
 * checks if a python object is a particle, and returns the
 * corresponding particle pointer, NULL otherwise
 */
CAPI_FUNC(MxParticle*) MxParticle_Get(MxParticleHandle *pypart);


/**
 * internal function that absoltuly has to be moved to a util file,
 * TODO: quick hack putting it here.
 *
 * points in a random direction with given magnitide mean, std
 */
MxVector3f MxRandomVector(float mean, float std);
MxVector3f MxRandomUnitVector();


/**
 * simple fission,
 *
 * divides a particle into two, and creates a new daughter particle in the
 * universe.
 *
 * Vector of numbers indicate how to split the attached chemical cargo.
 */
CAPI_FUNC(MxParticleHandle*) MxParticle_FissionSimple(MxParticle *part,
        MxParticleType *a, MxParticleType *b,
        int nPartitionRatios, float *partitionRations);

CAPI_FUNC(MxParticleHandle*) MxParticle_New(MxParticleType *type, 
                                            MxVector3f *position=NULL,
                                            MxVector3f *velocity=NULL,
                                            int *clusterId=NULL);

/**
 * Change the type of one particle to another.
 *
 * removes the particle from it's current type's list of objects,
 * and adds it to the new types list.
 *
 * changes the type pointer in the C MxParticle, and also changes
 * the type pointer in the Python MxPyParticle handle.
 */
CAPI_FUNC(HRESULT) MxParticle_Become(MxParticle *part, MxParticleType *type);

/**
 * The the particle type type
 */
CAPI_DATA(unsigned int) *MxParticle_Colors;

// Returns 1 if a type has been registered, otherwise 0
CAPI_FUNC(HRESULT) MxParticleType_checkRegistered(MxParticleType *type);

/**
 * mandatory internal function to initalize the particle and particle types
 *
 * sets the engine.types[0] particle.
 *
 * The engine.types array is assumed to be allocated, but not initialized.
 */
HRESULT _MxParticle_init();

#endif // INCLUDE_PARTICLE_H_
