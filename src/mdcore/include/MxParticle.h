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
#include "MxParticleTypeList.h"
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
 * 
 * If you're building a model, you should probably instead be working with a 
 * MxParticleHandle. 
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
     * @brief Get a handle for this particle. 
     * 
     * @return struct MxParticleHandle* 
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

    MxParticle *particle(int i);

    // style pointer, set at object construction time.
    // may be re-set by users later.
    // the base particle type has a default style.
    NOMStyle *style;

    /**
     * pointer to state vector (optional)
     */
    struct MxStateVector *state_vector;

    MxVector3f global_position();

    void set_global_position(const MxVector3f& pos);
    
    /**
     * performs a self-verify, in debug mode raises assertion if not valid
     */
    bool verify();

    
    /**
     * @brief Cast to a cluster type. Limits casting to cluster by type. 
     * 
     * @return MxCluster* 
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
 * @brief A handle to a particle. 
 *
 * The engine allocates particle memory in blocks, and particle
 * values get moved around all the time, so their addresses change.
 *
 * The partlist is always ordered by id, i.e. partlist[id]  always
 * points to the same particle, even though that particle may move
 * from cell to cell.
 * 
 * This is a safe way to work with a particle. 
 */
struct CAPI_EXPORT MxParticleHandle {
    int id;
    int typeId;

    /**
     * @brief Gets the actual particle of this handle. 
     * 
     * @return MxParticle* 
     */
    MxParticle *part();

    /**
     * @brief Gets the particle type of this handle. 
     * 
     * @return MxParticleType* 
     */
    MxParticleType *type();

    MxParticleHandle() : id(0), typeId(0) {}
    MxParticleHandle(const int &id, const int &typeId) : id(id), typeId(typeId) {}

    virtual MxParticleHandle* fission();

    /**
     * @brief Splits a single particle into two. Returns the new particle. 
     * 
     * @return MxParticleHandle* 
     */
    virtual MxParticleHandle* split();

    /**
     * @brief Destroys the particle, and removes it from inventory. 
     * 
     * Subsequent references to a destroyed particle result in an error. 
     * 
     * @return HRESULT 
     */
    HRESULT destroy();

    /**
     * @brief Calculates the particle's coordinates in spherical coordinates. 
     * 
     * By default, calculations are made with respect to the center of the universe. 
     * 
     * @param particle a particle to use as the origin, optional
     * @param origin a point to use as the origin, optional
     * @return MxVector3f 
     */
    MxVector3f sphericalPosition(MxParticle *particle=NULL, MxVector3f *origin=NULL);

    /**
     * @brief Computes the relative position with respect to an origin while 
     * optionally account for boundary conditions. 
     * 
     * @param origin origin
     * @param comp_bc flag to compensate for boundary conditions; default true
     * @return MxVector3f relative position with respect to the given origin
     */
    MxVector3f relativePosition(const MxVector3f &origin, const bool &comp_bc=true);

    /**
     * @brief Computes the virial tensor. Optionally pass a distance to include a neighborhood. 
     * 
     * @param radius A distance to define a neighborhood, optional
     * @return MxMatrix3f 
     */
    virtual MxMatrix3f virial(float *radius=NULL);

    /**
     * @brief Dynamically changes the *type* of an object. We can change the type of a 
     * MxParticleType-derived object to anyther pre-existing MxParticleType-derived 
     * type. What this means is that if we have an object of say type 
     * *A*, we can change it to another type, say *B*, and and all of the forces 
     * and processes that acted on objects of type A stip and the forces and 
     * processes defined for type B now take over. 
     * 
     * @param type new particle type
     * @return HRESULT 
     */
    HRESULT become(MxParticleType *type);

    /**
     * @brief Gets a list of nearby particles. 
     * 
     * @param distance optional search distance; default is simulation cutoff
     * @param types optional list of particle types to search by; default is all types
     * @return MxParticleList* 
     */
    MxParticleList *neighbors(const float *distance=NULL, const std::vector<MxParticleType> *types=NULL);

    /**
     * @brief Gets a list of all bonded neighbors. 
     * 
     * @return MxParticleList* 
     */
    MxParticleList *getBondedNeighbors();

    /**
     * @brief Calculates the distance to another particle
     * 
     * @param _other another particle. 
     * @return float 
     */
    float distance(MxParticleHandle *_other);
    std::vector<MxBondHandle*> *getBonds();

    double getCharge();
    void setCharge(const double &charge);
    double getMass();
    void setMass(const double &mass);
    bool getFrozen();
    void setFrozen(const bool frozen);
    bool getFrozenX();
    void setFrozenX(const bool frozen);
    bool getFrozenY();
    void setFrozenY(const bool frozen);
    bool getFrozenZ();
    void setFrozenZ(const bool frozen);
    NOMStyle *getStyle();
    void setStyle(NOMStyle *style);
    double getAge();
    double getRadius();
    void setRadius(const double &radius);
    std::string getName();
    std::string getName2();
    MxVector3f getPosition();
    void setPosition(MxVector3f position);
    MxVector3f getVelocity();
    void setVelocity(MxVector3f velocity);
    MxVector3f getForce();
    void setForce(MxVector3f force);
    int getId();
    int16_t getTypeId();
    uint16_t getFlags();
    MxStateVector *getSpecies();

    /**
     * Limits casting to cluster by type
     */
    operator MxClusterParticleHandle*();
};

/**
 * @brief Structure containing information on each particle type.
 *
 * This is only a definition for a *type* of particle, and not an 
 * actual particle with position, velocity, etc. However, particles 
 * of this *type* can be created with one of these. 
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

    /**
     * @brief Default mass of particles
     */
    double mass;
    
    double imass;
    
    /**
     * @brief Default charge of particles
     */
    double charge;

    /**
     * @brief Default radius of particles
     */
    double radius;

    /**
     * @brief Kinetic energy of all particles of this type. 
     */
    double kinetic_energy;

    /**
     * @brief Potential energy of all particles of this type. 
     */
    double potential_energy;

    /**
     * @brief Target energy of all particles of this type. 
     */
    double target_energy;

    /**
     * @brief Default minimum radius of this type. 
     * 
     * If a split event occurs, resulting particles will have a radius 
     * at least as great as this value. 
     */
    double minimum_radius;

    /** Nonbonded interaction parameters. */
    double eps, rmin;

    /**
     * @brief Default dynamics of particles of this type. 
     */
    unsigned char dynamics;

    /**
     * @brief Name of this particle type.
     */
    char name[MAX_NAME];
    
    char name2[MAX_NAME];

    /**
     * @brief list of particles that belong to this type.
     */
    MxParticleList parts;

    /**
     * @brief list of particle types that belong to this type.
     */
    MxParticleTypeList types;

    /**
     * @brief style pointer, optional.
     */
    NOMStyle *style;

    /**
     * @brief optional pointer to species list. This is the metadata for the species
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
     * @brief get the i'th particle that's a member of this type.
     * 
     * @param i index of particle to get
     * @return MxParticle* 
     */
    MxParticle *particle(int i);

    /**
     * @brief Get all current particle type ids, excluding clusters
     * 
     * @return std::set<int> 
     */
    static std::set<short int> particleTypeIds();

    bool isCluster();

    /**
     * @brief Particle constructor
     * 
     * @param position position of new particle, optional
     * @param velocity velocity of new particle, optional
     * @param clusterId id of parent cluster, optional
     * @return MxParticleHandle* 
     */
    MxParticleHandle *operator()(MxVector3f *position=NULL,
                                 MxVector3f *velocity=NULL,
                                 int *clusterId=NULL);

    /**
     * @brief Particle type constructor. 
     * 
     * New type is constructed from the definition of the calling type. 
     * 
     * @param _name name of the new type
     * @return MxParticleType* 
     */
    MxParticleType* newType(const char *_name);

    /**
     * @brief Registers a type with the engine.
     * 
     * Note that this occurs automatically, unless noReg==true in constructor.  
     * 
     * @return HRESULT 
     */
    virtual HRESULT registerType();

    /**
     * @brief A callback for when a type is registered
     */
    virtual void on_register() {}

    /**
     * @brief Tests whether this type is registered
     * 
     * @return true if registered
     */
    bool isRegistered();

    /**
     * @brief Get the type engine instance
     * 
     * @return MxParticleType* 
     */
    virtual MxParticleType *get();

    MxParticleType(const bool &noReg=false);
    virtual ~MxParticleType() {}

    bool getFrozen();
    void setFrozen(const bool &frozen);
    bool getFrozenX();
    void setFrozenX(const bool &frozen);
    bool getFrozenY();
    void setFrozenY(const bool &frozen);
    bool getFrozenZ();
    void setFrozenZ(const bool &frozen);
    // temperature is an ensemble property
    double getTemperature();
    double getTargetTemperature();

    /**
     * @brief Get all particles of this type. 
     * 
     * @return MxParticleList* 
     */
    MxParticleList *items();
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
