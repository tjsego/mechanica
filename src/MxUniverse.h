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

struct CAPI_EXPORT MxUniverse  {

    static MxVector3f origin();

    static  MxVector3f dim();


    bool isRunning;

    // name of the model / script, usually picked up from command line;
    std::string name;
    
    static MxMatrix3f *virial(MxVector3f *origin=NULL, float *radius=NULL, std::vector<MxParticleType*> *types=NULL);
    static MxVector3f getCenter();

    static HRESULT step(const double &until=0, const double &dt=0);
    static HRESULT stop();
    static HRESULT start();
    static HRESULT reset();
    static MxParticleList *particles();
    static void resetSpecies();
    static std::vector<std::vector<std::vector<MxParticleList*> > > grid(MxVector3i shape);
    static std::vector<MxBondHandle*> *bonds();

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
