/*
 * MxUniverse.h
 *
 *  Created on: Sep 7, 2019
 *      Author: andy
 */

#ifndef SRC_MXUNIVERSE_H_
#define SRC_MXUNIVERSE_H_

#include "mechanica_private.h"
#include "mdcore_single.h"
#include <unordered_map>

/**
 * @brief The universe is a top level singleton object, and is automatically
 * initialized when the simulator loads. The universe is a representation of the
 * physical universe that we are simulating, and is the repository for all
 * physical object representations.
 * 
 * All properties and methods on the universe are static, and you never actually
 * instantiate a universe.
 * 
 * Universe has a variety of properties such as boundary conditions, and stores
 * all the physical objects such as particles, bonds, potentials, etc.
 */
struct CAPI_EXPORT MxUniverse  {

    /**
     * @brief Gets the origin of the universe
     * 
     * @return MxVector3f 
     */
    static MxVector3f origin();

    /**
     * @brief Gets the dimensions of the universe
     * 
     * @return MxVector3f 
     */
    static  MxVector3f dim();


    bool isRunning;

    // name of the model / script, usually picked up from command line;
    std::string name;
    
    /**
     * @brief Computes the virial tensor for the either the entire simulation 
     * domain, or a specific local virial tensor at a location and 
     * radius. Optionally can accept a list of particle types to restrict the 
     * virial calculation for specify types.
     * 
     * @param origin An optional length-3 array for the origin. Defaults to the center of the simulation domain if not given.
     * @param radius An optional number specifying the size of the region to compute the virial tensor for. Defaults to the entire simulation domain.
     * @param types An optional list of :class:`Particle` types to include in the calculation. Defaults to every particle type.
     * @return MxMatrix3f* 
     */
    static MxMatrix3f *virial(MxVector3f *origin=NULL, float *radius=NULL, std::vector<MxParticleType*> *types=NULL);

    static MxVector3f getCenter();

    /**
     * @brief Performs a single time step ``dt`` of the universe if no arguments are 
     * given. Optionally runs until ``until``, and can use a different timestep 
     * of ``dt``.
     * 
     * @param until runs the timestep for this length of time, optional.
     * @param dt overrides the existing time step, and uses this value for time stepping, optional.
     * @return HRESULT 
     */
    static HRESULT step(const double &until=0, const double &dt=0);

    /**
     * @brief Stops the universe time evolution. This essentially freezes the universe, 
     * everything remains the same, except time no longer moves forward.
     * 
     * @return HRESULT 
     */
    static HRESULT stop();

    /**
     * @brief Starts the universe time evolution, and advanced the universe forward by 
     * timesteps in ``dt``. All methods to build and manipulate universe objects 
     * are valid whether the universe time evolution is running or stopped.
     * 
     * @return HRESULT 
     */
    static HRESULT start();

    static HRESULT reset();

    /**
     * @brief Gets all particles in the universe
     * 
     * @return MxParticleList* 
     */
    static MxParticleList *particles();
    static void resetSpecies();

    /**
     * @brief Gets a three-dimesional array of particle lists, of all the particles in the system. 
     * 
     * @param shape shape of grid
     * @return std::vector<std::vector<std::vector<MxParticleList*> > > 
     */
    static std::vector<std::vector<std::vector<MxParticleList*> > > grid(MxVector3i shape);

    /**
     * @brief Get all bonds in the universe
     * 
     * @return std::vector<MxBondHandle*>* 
     */
    static std::vector<MxBondHandle*> *bonds();

    /**
     * @brief Get all angles in the universe
     * 
     * @return std::vector<MxAngleHandle*>* 
     */
    static std::vector<MxAngleHandle*> *angles();

    /**
     * @brief Get all dihedrals in the universe
     * 
     * @return std::vector<MxDihedral*>* 
     */
    static std::vector<MxDihedralHandle*> *dihedrals();

    /**
     * @brief Get the universe temperature. 
     * 
     * The universe can be run with, or without a thermostat. With a thermostat, 
     * getting / setting the temperature changes the temperature that the thermostat 
     * will try to keep the universe at. When the universe is run without a 
     * thermostat, reading the temperature returns the computed universe temp, but 
     * attempting to set the temperature yields an error. 
     * 
     * @return double 
     */
    double getTemperature();
    double getTime();
    double getDt();
    MxEventList *getEventList();
    MxBoundaryConditions *getBoundaryConditions();
    double getKineticEnergy();
    int getNumTypes();
};

/**
 *
 * @brief Initialize an #engine with the given data.
 *
 * The number of spatial cells in each cartesion dimension is floor( dim[i] / L[i] ), or
 * the physical size of the space in that dimension divided by the minimum size size of
 * each cell.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 *
 * @param L The minimum spatial cell edge length in each dimension.
 *
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * int engine_init ( struct engine *e , const double *origin , const double *dim , double *L ,
        double cutoff , unsigned int period , int max_type , unsigned int flags );
 */

struct CAPI_EXPORT MxUniverseConfig {
    MxVector3f origin;
    MxVector3f dim;
    MxVector3i spaceGridSize;
    double cutoff;
    uint32_t flags;
    uint32_t maxTypes;
    double dt;
    double temp;
    int nParticles;
    int threads;
    EngineIntegrator integrator;
    
    // pointer to boundary conditions ctor data
    // these objects are parsed initializing the engine.
    MxBoundaryConditionsArgsContainer *boundaryConditionsPtr;

    double max_distance;
    
    
    // bitmask of timers to show in performance counter output.
    uint32_t timers_mask;
    
    long timer_output_period;
    
    MxUniverseConfig();
    
    // just set the object, borow a pointer to python handle
    void setBoundaryConditions(MxBoundaryConditionsArgsContainer *_bcArgs) {
        boundaryConditionsPtr = _bcArgs;
    }
    
    ~MxUniverseConfig() {
        if(boundaryConditionsPtr) {
            delete boundaryConditionsPtr;
            boundaryConditionsPtr = 0;
        }
    }
};


/**
 * runs the universe a pre-determined period of time, until.
 * can use micro time steps of 'dt' which override the
 * saved universe dt.
 *
 * if until is 0, it is ignored and the universe.dt is used.
 * if dt is 0, it is ignored, and the universe.dt is used as
 * a single time step.
 */
CAPI_FUNC(HRESULT) MxUniverse_Step(double until, double dt);


/**
 * starts the universe time evolution. The simulator
 * actually advances the universe, this method just
 * tells the simulator to perform the time evolution.
 */
enum MxUniverse_Flags {
    MX_RUNNING = 1 << 0,

    MX_SHOW_PERF_STATS = 1 << 1,

    // in ipython message loop, monitor console
    MX_IPYTHON_MSGLOOP = 1 << 2,

    // standard polling message loop
    MX_POLLING_MSGLOOP = 1 << 3,
};

/**
 * get a flag value
 */
CAPI_FUNC(int) MxUniverse_Flag(MxUniverse_Flags flag);

/**
 * sets / clears a flag value
 */
CAPI_FUNC(HRESULT) MxUniverse_SetFlag(MxUniverse_Flags flag, int value);



/**
 * The single global instance of the universe
 */
CAPI_DATA(MxUniverse) Universe;

/**
 * Universe instance accessor
 * 
 */
CAPI_FUNC(MxUniverse*) getUniverse();

#endif /* SRC_MXUNIVERSE_H_ */
