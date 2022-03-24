/*
 * MxSimulator.cpp
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#include <MxSimulator.h>
#include <rendering/MxUI.h>
#include <rendering/MxTestView.h>

#include <Magnum/GL/Context.h>

#include "rendering/MxApplication.h"
#include "rendering/MxUniverseRenderer.h"
#include <rendering/MxGlfwApplication.h>
#include <rendering/MxWindowlessApplication.h>
#include <rendering/MxClipPlane.hpp>
#include <map>
#include <sstream>
#include <MxUniverse.h>
#include <MxSystem.h>
#include <MxCluster.hpp>
#include <MxLogger.h>
#include <mx_error.h>
#include <mx_parse.h>
#include <io/MxFIO.h>

// mdcore errs.h
#include <errs.h>

#include <thread>

/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
#define ENGINE_FLAGS engine_flag_none
#endif
#ifndef CPU_TPS
#define CPU_TPS 2.67e+9
#endif


#define SIM_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = MX_FUNCTION; \
            err += "universe not initialized"; \
            mx_exp(std::domain_error(err.c_str())); \
        }

#define SIM_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        mx_exp(e); return retval; \
    }

static MxSimulator* Simulator = NULL;
#ifdef MX_WITHCUDA
static MxSimulatorCUDAConfig *SimulatorCUDAConfig = NULL;
#endif
static bool isTerminalInteractiveShell = false;

static void simulator_interactive_run();


MxSimulator_Config::MxSimulator_Config():
            _title{"Mechanica Application"},
            _size{800, 600},
            _dpiScalingPolicy{MxSimulator_DpiScalingPolicy::Default},
            queues{4},
           _windowless{ false }
{
    _windowFlags = MxSimulator::WindowFlags::Resizable |
                   MxSimulator::WindowFlags::Focused   |
                   MxSimulator::WindowFlags::Hidden;  // make the window initially hidden
}



MxSimulator::GLConfig::GLConfig():
_colorBufferSize{8, 8, 8, 0}, _depthBufferSize{24}, _stencilBufferSize{0},
_sampleCount{0}, _version{GL::Version::None},
#ifndef MAGNUM_TARGET_GLES
_flags{Flag::ForwardCompatible},
#else
_flags{},
#endif
_srgbCapable{false} {}

MxSimulator::GLConfig::~GLConfig() = default;




#define SIMULATOR_CHECK()  if (!Simulator) { return mx_error(E_INVALIDARG, "Simulator is not initialized"); }

/**
 * Make a Arguments struct from a string list,
 * Magnum has different args for different app types,
 * so this needs to be a template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(const std::vector<std::string> &args) {

        for(auto &a : args) {
            strings.push_back(a);
            cstrings.push_back(a.c_str());

            Log(LOG_INFORMATION) <<  "args: " << a ;;
        }

        // int reference, keep an ivar around for it to point to.
        argsIntReference = cstrings.size();
        char** constRef = const_cast<char**>(cstrings.data());

        pArgs = new T(argsIntReference, constRef);
    }

    ~ArgumentsWrapper() {
        delete pArgs;
    }

    std::vector<std::string> strings;
    std::vector<const char*> cstrings;
    T *pArgs = NULL;
    int argsIntReference;
};

static HRESULT initSimConfigFromFile(const std::string &loadFilePath, MxSimulator_Config &conf) {

    if(MxFIO::currentRootElement != NULL) {
        mx_error(E_FAIL, "Cannot load from multiple files");
        return E_FAIL;
    }

    MxIOElement *fe = MxFIO::fromFile(loadFilePath);
    if(fe == NULL) {
        mx_error(E_FAIL, "Error loading file");
        return E_FAIL;
    }

    MxMetaData metaData, metaDataFile;

    auto feItr = fe->children.find(MxFIO::KEY_METADATA);
    if(feItr == fe->children.end() || mx::io::fromFile(*feItr->second, metaData, &metaDataFile) != S_OK) {
        mx_error(E_FAIL, "Error loading metadata");
        return E_FAIL;
    }

    feItr = fe->children.find(MxFIO::KEY_SIMULATOR);
    if(feItr == fe->children.end() || mx::io::fromFile(*feItr->second, metaDataFile, &conf) != S_OK) {
        mx_error(E_FAIL, "Error loading simulator");
        return E_FAIL;
    }

    conf.importDataFilePath = new std::string(loadFilePath);

    return S_OK;
}

static void parse_kwargs(MxSimulator_Config &conf, 
                         MxVector3f *dim=NULL, 
                         double *cutoff=NULL, 
                         MxVector3i *cells=NULL, 
                         unsigned *threads=NULL, 
                         int *integrator=NULL, 
                         double *dt=NULL, 
                         int *bcValue=NULL, 
                         std::unordered_map<std::string, unsigned int> *bcVals=NULL, 
                         std::unordered_map<std::string, MxVector3f> *bcVels=NULL, 
                         std::unordered_map<std::string, float> *bcRestores=NULL, 
                         MxBoundaryConditionsArgsContainer *bcArgs=NULL, 
                         double *max_distance=NULL, 
                         bool *windowless=NULL, 
                         MxVector2i *window_size=NULL, 
                         unsigned int *seed=NULL, 
                         uint32_t *perfcounters=NULL, 
                         int *perfcounter_period=NULL, 
                         int *logger_level=NULL, 
                         std::vector<std::tuple<MxVector3f, MxVector3f> > *clip_planes=NULL) 
{
    if(dim) conf.universeConfig.dim = *dim;
    if(cutoff) conf.universeConfig.cutoff = *cutoff;
    if(cells) conf.universeConfig.spaceGridSize = *cells;
    if(threads) conf.universeConfig.threads = *threads;
    if(integrator) {
        int kind = *integrator;
        switch (kind) {
            case FORWARD_EULER:
            case RUNGE_KUTTA_4:
                conf.universeConfig.integrator = (EngineIntegrator)kind;
                break;
            default: {
                std::string msg = "invalid integrator kind: ";
                msg += std::to_string(kind);
                mx_exp(std::logic_error(msg));
            }
        }
    }
    if(dt) conf.universeConfig.dt = *dt;

    if(bcArgs) conf.universeConfig.setBoundaryConditions(bcArgs);
    else conf.universeConfig.setBoundaryConditions(new MxBoundaryConditionsArgsContainer(bcValue, bcVals, bcVels, bcRestores));
    
    if(max_distance) conf.universeConfig.max_distance = *max_distance;
    if(windowless) conf.setWindowless(*windowless);
    if(window_size) conf.setWindowSize(*window_size);
    if(seed) conf.setSeed(*seed);
    if(perfcounters) conf.universeConfig.timers_mask = *perfcounters;
    if(perfcounter_period) conf.universeConfig.timer_output_period = *perfcounter_period;
    if(logger_level) MxLogger::setLevel(*logger_level);
    if(clip_planes) conf.clipPlanes = MxParsePlaneEquation(*clip_planes);
}

// intermediate kwarg parsing
static void parse_kwargs(const std::vector<std::string> &kwargs, MxSimulator_Config &conf) {

    Log(LOG_INFORMATION) << "parsing vector string input";

    std::string s;

    if(mx::parse::has_kwarg(kwargs, "load_file")) {
        s = mx::parse::kwargVal(kwargs, "load_file");

        Log(LOG_INFORMATION) << "got load file: " << s;
        
        if(initSimConfigFromFile(s, conf) != S_OK) 
            return;
    }

    MxVector3f *dim;
    if(mx::parse::has_kwarg(kwargs, "dim")) {
        s = mx::parse::kwargVal(kwargs, "dim");
        dim = new MxVector3f(mx::parse::strToVec<float>(s));

        Log(LOG_INFORMATION) << "got dim: " 
                             << std::to_string(dim->x()) << "," 
                             << std::to_string(dim->y()) << "," 
                             << std::to_string(dim->z());
    }
    else dim = NULL;

    double *cutoff;
    if(mx::parse::has_kwarg(kwargs, "cutoff")) {
        s = mx::parse::kwargVal(kwargs, "cutoff");
        cutoff = new double(mx::cast<std::string, double>(s));

        Log(LOG_INFORMATION) << "got cutoff: " << std::to_string(*cutoff);
    }
    else cutoff = NULL;

    MxVector3i *cells;
    if(mx::parse::has_kwarg(kwargs, "cells")) {
        s = mx::parse::kwargVal(kwargs, "cells");
        cells = new MxVector3i(mx::parse::strToVec<int>(s));

        Log(LOG_INFORMATION) << "got cells: " 
                             << std::to_string(cells->x()) << "," 
                             << std::to_string(cells->y()) << "," 
                             << std::to_string(cells->z());
    }
    else cells = NULL;

    unsigned *threads;
    if(mx::parse::has_kwarg(kwargs, "threads")) {
        s = mx::parse::kwargVal(kwargs, "threads");
        threads = new unsigned(mx::cast<std::string, unsigned>(s));

        Log(LOG_INFORMATION) << "got threads: " << std::to_string(*threads);
    }
    else threads = NULL;

    int *integrator;
    if(mx::parse::has_kwarg(kwargs, "integrator")) {
        s = mx::parse::kwargVal(kwargs, "integrator");
        integrator = new int(mx::cast<std::string, int>(s));

        Log(LOG_INFORMATION) << "got integrator: " << std::to_string(*integrator);
    }
    else integrator = NULL;

    double *dt;
    if(mx::parse::has_kwarg(kwargs, "dt")) {
        s = mx::parse::kwargVal(kwargs, "dt");
        dt = new double(mx::cast<std::string, double>(s));

        Log(LOG_INFORMATION) << "got dt: " << std::to_string(*dt);
    }
    else dt = NULL;

    MxBoundaryConditionsArgsContainer *bcArgs;
    if(mx::parse::has_mapKwarg(kwargs, "bc")) {
        // example: 
        // bc={left={velocity={x=0;y=2};restore=1.0};bottom={type=no_slip}}
        s = mx::parse::kwargVal(kwargs, "bc");
        std::vector<std::string> mapEntries = mx::parse::mapStrToStrVec(mx::parse::mapStrip(s));

        std::unordered_map<std::string, unsigned int> *bcVals = new std::unordered_map<std::string, unsigned int>();
        std::unordered_map<std::string, MxVector3f> *bcVels = new std::unordered_map<std::string, MxVector3f>();
        std::unordered_map<std::string, float> *bcRestores = new std::unordered_map<std::string, float>();

        std::string name;
        std::vector<std::string> val;
        for(auto &ss : mapEntries) {
            std::tie(name, val) = mx::parse::kwarg_getNameMapVals(ss);

            if(mx::parse::has_kwarg(val, "type")) {
                std::string ss = mx::parse::kwargVal(val, "type");
                (*bcVals)[name] = MxBoundaryConditions::boundaryKindFromString(ss);

                Log(LOG_INFORMATION) << "got bc type: " << name << "->" << ss;
            }
            else if(mx::parse::has_kwarg(val, "velocity")) {
                std::string ss = mx::parse::mapStrip(mx::parse::kwarg_strMapVal(val, "velocity"));
                std::vector<std::string> sv = mx::parse::mapStrToStrVec(ss);
                float x, y, z;
                if(mx::parse::has_kwarg(sv, "x")) x = mx::cast<std::string, float>(mx::parse::kwargVal(sv, "x"));
                else x = 0.0;
                if(mx::parse::has_kwarg(sv, "y")) y = mx::cast<std::string, float>(mx::parse::kwargVal(sv, "y"));
                else y = 0.0;
                if(mx::parse::has_kwarg(sv, "z")) z = mx::cast<std::string, float>(mx::parse::kwargVal(sv, "z"));
                else z = 0.0;

                auto vel = MxVector3f(x, y, z);
                (*bcVels)[name] = vel;

                Log(LOG_INFORMATION) << "got bc velocity: " << name << "->" << vel;

                if(mx::parse::has_kwarg(val, "restore")) {
                    std::string ss = mx::parse::kwargVal(val, "restore");
                    (*bcRestores)[name] = mx::cast<std::string, float>(ss);

                    Log(LOG_INFORMATION) << "got bc restore: " << name << "->" << ss;
                }
            }
        }
        
        bcArgs = new MxBoundaryConditionsArgsContainer(NULL, bcVals, bcVels, bcRestores);
    }
    else if(mx::parse::has_kwarg(kwargs, "bc")) {
        // example: 
        // bc=no_slip
        s = mx::parse::kwargVal(kwargs, "bc");
        int *bcValue = new int(MxBoundaryConditions::boundaryKindFromString(s));
        bcArgs = new MxBoundaryConditionsArgsContainer(bcValue, NULL, NULL, NULL);

        Log(LOG_INFORMATION) << "got bc val: " << std::to_string(*bcValue);
    }
    else bcArgs = NULL;

    double *max_distance;
    if(mx::parse::has_kwarg(kwargs, "max_distance")) {
        s = mx::parse::kwargVal(kwargs, "max_distance");
        max_distance = new double(mx::cast<std::string, double>(s));

        Log(LOG_INFORMATION) << "got max_distance: " << std::to_string(*max_distance);
    }
    else max_distance = NULL;

    bool *windowless;
    if(mx::parse::has_kwarg(kwargs, "windowless")) {
        s = mx::parse::kwargVal(kwargs, "windowless");
        windowless = new bool(mx::cast<std::string, bool>(s));

        Log(LOG_INFORMATION) << "got windowless: " << (*windowless ? "True" : "False");
    }
    else windowless = NULL;

    MxVector2i *window_size;
    if(mx::parse::has_kwarg(kwargs, "window_size")) {
        s = mx::parse::kwargVal(kwargs, "window_size");
        window_size = new MxVector2i(mx::parse::strToVec<int>(s));

        Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

    unsigned int *seed;
    if(mx::parse::has_kwarg(kwargs, "seed")) {
        s = mx::parse::kwargVal(kwargs, "seed");
        seed = new unsigned int(mx::cast<std::string, unsigned int>(s));

        Log(LOG_INFORMATION) << "got seed: " << std::to_string(*seed);
    }
    else seed = NULL;

    uint32_t *perfcounters;
    if(mx::parse::has_kwarg(kwargs, "perfcounters")) {
        s = mx::parse::kwargVal(kwargs, "perfcounters");
        perfcounters = new uint32_t(mx::cast<std::string, uint32_t>(s));

        Log(LOG_INFORMATION) << "got perfcounters: " << std::to_string(*perfcounters);
    }
    else perfcounters = NULL;

    int *perfcounter_period;
    if(mx::parse::has_kwarg(kwargs, "perfcounter_period")) {
        s = mx::parse::kwargVal(kwargs, "perfcounter_period");
        perfcounter_period = new int(mx::cast<std::string, int>(s));

        Log(LOG_INFORMATION) << "got perfcounter_period: " << std::to_string(*perfcounter_period);
    }
    else perfcounter_period = NULL;
    
    int *logger_level;
    if(mx::parse::has_kwarg(kwargs, "logger_level")) {
        s = mx::parse::kwargVal(kwargs, "logger_level");
        logger_level = new int(mx::cast<std::string, int>(s));

        Log(LOG_INFORMATION) << "got logger_level: " << std::to_string(*logger_level);
    }
    else logger_level = NULL;
    
    std::vector<std::tuple<MxVector3f, MxVector3f> > *clip_planes;
    MxVector3f point, normal;
    if(mx::parse::has_kwarg(kwargs, "clip_planes")) {
        // ex: clip_planes=0,1,2,3,4,5;1,2,3,4,5,6
        clip_planes = new std::vector<std::tuple<MxVector3f, MxVector3f> >();

        s = mx::parse::kwargVal(kwargs, "clip_planes");
        std::vector<std::string> sc = mx::parse::mapStrToStrVec(s);
        for (auto &ss : sc) {
            std::vector<float> svec = mx::parse::strToVec<float>(ss);
            point = MxVector3f(svec[0], svec[1], svec[2]);
            normal = MxVector3f(svec[3], svec[4], svec[5]);
            clip_planes->push_back(std::make_tuple(point, normal));

            Log(LOG_INFORMATION) << "got clip plane: " << point << ", " << normal;
        }
    }
    else clip_planes = NULL;
    parse_kwargs(conf, 
                 dim, 
                 cutoff, 
                 cells, 
                 threads, 
                 integrator, 
                 dt, 
                 NULL, NULL, NULL, NULL, 
                 bcArgs, 
                 max_distance, 
                 windowless, 
                 window_size, 
                 seed, 
                 perfcounters, 
                 perfcounter_period, 
                 logger_level, 
                 clip_planes);
}

// (5) Initializer list constructor
const std::map<std::string, int> configItemMap {
    {"none", MXSIMULATOR_NONE},
    {"windowless", MXSIMULATOR_WINDOWLESS},
    {"glfw", MXSIMULATOR_GLFW}
};

#define MX_CLASS METH_CLASS | METH_VARARGS | METH_KEYWORDS


CAPI_FUNC(HRESULT) MxSimulator::pollEvents()
{
    SIMULATOR_CHECK();
    return Simulator->app->pollEvents();
}

HRESULT MxSimulator::waitEvents()
{
    SIMULATOR_CHECK();
    return Simulator->app->waitEvents();
}

HRESULT MxSimulator::waitEventsTimeout(double timeout)
{
    SIMULATOR_CHECK();
    return Simulator->app->waitEventsTimeout(timeout);
}

HRESULT MxSimulator::postEmptyEvent()
{
    SIMULATOR_CHECK();
    return Simulator->app->postEmptyEvent();
}

HRESULT MxSimulator::swapInterval(int si)
{
    SIMULATOR_CHECK();
    return Simulator->app->setSwapInterval(si);
}

const int MxSimulator::getNumThreads() {
    SIM_TRY();
    return _Engine.nr_runners;
    SIM_FINALLY(0);
}

const MxGlfwWindow *MxSimulator::getWindow() {
    SIM_TRY();
    return Simulator->app->getWindow();
    SIM_FINALLY(0);
}

#ifdef MX_WITHCUDA
MxSimulatorCUDAConfig *MxSimulator::getCUDAConfig() {
    return SimulatorCUDAConfig;
}

HRESULT MxSimulator::makeCUDAConfigCurrent(MxSimulatorCUDAConfig *config) {
    if(SimulatorCUDAConfig) {
        mx_exp(std::domain_error("Error, Simulator is already initialized" ));
        return E_FAIL;
    }
    SimulatorCUDAConfig = config;
    return S_OK;
}
#endif

bool Mx_TerminalInteractiveShell() {
    return isTerminalInteractiveShell;
}

HRESULT Mx_setTerminalInteractiveShell(const bool &_interactive) {
    isTerminalInteractiveShell = _interactive;
    return S_OK;
}

HRESULT modules_init() {
    Log(LOG_DEBUG) << ", initializing modules... " ;

    _MxParticle_init();
    _MxCluster_init();

    return S_OK;
}

int universe_init(const MxUniverseConfig &conf ) {

    Universe.events = new MxEventBaseList();

    MxVector3i cells = conf.spaceGridSize;

    double cutoff = conf.cutoff;

    int nr_runners = conf.threads;

    double _origin[3];
    double _dim[3];
    for(int i = 0; i < 3; ++i) {
        _origin[i] = conf.origin[i];
        _dim[i] = conf.dim[i];
    }


    Log(LOG_INFORMATION) << "main: initializing the engine... ";
    
    if ( engine_init( &_Engine , _origin , _dim , cells.data() , cutoff , conf.boundaryConditionsPtr ,
            conf.maxTypes , engine_flag_none ) != 0 ) {
        MX_RETURN_EXP(std::runtime_error(errs_getstring(0)));
    }

    _Engine.dt = conf.dt;
    _Engine.time = conf.start_step;
    _Engine.temperature = conf.temp;
    _Engine.integrator = conf.integrator;

    _Engine.timers_mask = conf.timers_mask;
    _Engine.timer_output_period = conf.timer_output_period;

    if(conf.max_distance >= 0) {
        // max_velocity is in absolute units, convert
        // to scale fraction.

        _Engine.particle_max_dist_fraction = conf.max_distance / _Engine.s.h[0];
    }

    const char* inte = NULL;

    switch(_Engine.integrator) {
    case EngineIntegrator::FORWARD_EULER:
        inte = "Forward Euler";
        break;
    case EngineIntegrator::RUNGE_KUTTA_4:
        inte = "Ruge-Kutta-4";
        break;
    }

    Log(LOG_INFORMATION) << "engine integrator: " << inte;
    Log(LOG_INFORMATION) << "engine: n_cells: " << _Engine.s.nr_cells << ", cell width set to " << cutoff;
    Log(LOG_INFORMATION) << "engine: cell dimensions = [" << _Engine.s.cdim[0] << ", " << _Engine.s.cdim[1] << ", " << _Engine.s.cdim[2] << "]";
    Log(LOG_INFORMATION) << "engine: cell size = [" << _Engine.s.h[0]  << ", " <<_Engine.s.h[1] << ", " << _Engine.s.h[2] << "]";
    Log(LOG_INFORMATION) << "engine: cutoff set to " << cutoff;
    Log(LOG_INFORMATION) << "engine: nr tasks: " << _Engine.s.nr_tasks;
    Log(LOG_INFORMATION) << "engine: nr cell pairs: " <<_Engine.s.nr_pairs;
    Log(LOG_INFORMATION) << "engine: dt: " << _Engine.dt;
    Log(LOG_INFORMATION) << "engine: max distance fraction: " << _Engine.particle_max_dist_fraction;

    // start the engine

    if ( engine_start( &_Engine , nr_runners , nr_runners ) != 0 ) {
        Log(LOG_ERROR) << errs_getstring(0);
        mx_exp(std::runtime_error(errs_getstring(0)));
    }

    if ( modules_init() != S_OK ) {
        Log(LOG_ERROR) << errs_getstring(0);
        mx_exp(std::runtime_error(errs_getstring(0)));
    }

    // if loading from file, populate universe if data is available
    
    if(MxFIO::currentRootElement != NULL) {
        Log(LOG_INFORMATION) << "Populating universe from file";

        MxMetaData metaData, metaDataFile;

        auto feItr = MxFIO::currentRootElement->children.find(MxFIO::KEY_METADATA);
        if(feItr == MxFIO::currentRootElement->children.end() || mx::io::fromFile(*feItr->second, metaData, &metaDataFile) != S_OK) {
            mx_error(E_FAIL, "Error loading metadata");
            return E_FAIL;
        }

        feItr = MxFIO::currentRootElement->children.find(MxFIO::KEY_UNIVERSE);
        if(feItr != MxFIO::currentRootElement->children.end()) {
            if(mx::io::fromFile(*feItr->second, metaDataFile, MxUniverse::get()) != S_OK) {
                mx_error(E_FAIL, "Error loading universe");
                return E_FAIL;
            }
        }
    }

    fflush(stdout);

    return 0;
}

HRESULT MxSimulator::run(double et)
{
    SIMULATOR_CHECK();

    Log(LOG_INFORMATION) <<  "simulator run(" << et << ")" ;

    return Simulator->app->run(et);
}

HRESULT MxSimulator_initC(const MxSimulator_Config &conf, const std::vector<std::string> &appArgv) {

    std::thread::id id = std::this_thread::get_id();
    Log(LOG_INFORMATION) << "thread id: " << id;

    try {

        if(Simulator) {
            mx_exp(std::domain_error("Error, Simulator is already initialized" ));
        }
        
        MxSimulator *sim = new MxSimulator();

        #ifdef MX_WITHCUDA
        MxCUDA::init();
        MxCUDA::setGLDevice(0);
        SimulatorCUDAConfig = new MxSimulatorCUDAConfig();
        #endif
        
        Universe.name = conf.title();

        Log(LOG_INFORMATION) << "got universe name: " << Universe.name;

        MxSetSeed(const_cast<MxSimulator_Config&>(conf).seed());

        // init the engine first
        /* Initialize scene particles */
        universe_init(conf.universeConfig);

        if(conf.windowless()) {
            Log(LOG_INFORMATION) <<  "creating Windowless app" ;
            
            ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(appArgv);

            MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

            if(FAILED(windowlessApp->createContext(conf))) {
                delete windowlessApp;

                mx_exp(std::domain_error("could not create windowless gl context"));
            }
            else {
                sim->app = windowlessApp;
            }

	    Log(LOG_TRACE) << "sucessfully created windowless app";
        }
        else {
            Log(LOG_INFORMATION) <<  "creating GLFW app" ;
            
            ArgumentsWrapper<MxGlfwApplication::Arguments> margs(appArgv);

            MxGlfwApplication *glfwApp = new MxGlfwApplication(*margs.pArgs);
            
            if(FAILED(glfwApp->createContext(conf))) {
                Log(LOG_DEBUG) << "deleting failed glfwApp";
                delete glfwApp;
                mx_exp(std::domain_error("could not create  gl context"));
            }
            else {
                sim->app = glfwApp;
            }
        }

        Log(LOG_INFORMATION) << "sucessfully created application";

        sim->makeCurrent();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSimulator_init(const std::vector<std::string> &argv) {

    try {

        MxSimulator_Config conf;
        
        if(argv.size() > 0) {
            std::string name = argv[0];
            conf.setTitle(name);
        }

        Log(LOG_INFORMATION) << "got universe name: " << Universe.name;
        
        // set default state of config
        conf.setWindowless(false);

        if(argv.size() > 1) {
            parse_kwargs(argv, conf);
        }

        Log(LOG_INFORMATION) << "successfully parsed args";

        return MxSimulator_initC(conf, argv);
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSimulator::show()
{
    SIMULATOR_CHECK();

    return Simulator->app->show();
}

HRESULT MxSimulator::redraw()
{
    SIMULATOR_CHECK();
    return Simulator->app->redraw();
}

HRESULT MxSimulator::initConfig(const MxSimulator_Config &conf, const MxSimulator::GLConfig &glConf)
{
    if(Simulator) {
        return mx_error(E_FAIL, "simulator already initialized");
    }

    MxSimulator *sim = new MxSimulator();

    #ifdef MX_WITHCUDA
    MxCUDA::init();
    MxCUDA::setGLDevice(0);
    SimulatorCUDAConfig = new MxSimulatorCUDAConfig();
    #endif

    // init the engine first
    /* Initialize scene particles */
    universe_init(conf.universeConfig);


    if(conf.windowless()) {

        /*



        MxWindowlessApplication::Configuration windowlessConf;

        MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

        if(!windowlessApp->tryCreateContext(conf)) {
            delete windowlessApp;

            mx_exp(std::domain_error("could not create windowless gl context"));
        }
        else {
            sim->app = windowlessApp;
        }
        */
    }
    else {

        Log(LOG_INFORMATION) <<  "creating GLFW app" ;;

        int argc = conf.argc;

        MxGlfwApplication::Arguments args{argc, conf.argv};

        MxGlfwApplication *glfwApp = new MxGlfwApplication(args);

        glfwApp->createContext(conf);

        sim->app = glfwApp;
    }

    Log(LOG_INFORMATION);

    sim->makeCurrent();

    return S_OK;
}

HRESULT MxSimulator::close()
{
    SIMULATOR_CHECK();
    return Simulator->app->close();
}

HRESULT MxSimulator::destroy()
{
    SIMULATOR_CHECK();
    return Simulator->app->destroy();
}

/**
 * gets the global simulator object, returns NULL if fail.
 */
MxSimulator *MxSimulator::get() {
    if(Simulator) {
        return Simulator;
    }
    Log(LOG_WARNING) << "Simulator is not initialized";
    return NULL;
}

HRESULT MxSimulator::makeCurrent() {
    if(Simulator) {
        mx_exp(std::logic_error("Simulator is already initialized"));
        return E_FAIL;
    }
    Simulator = this;
    return S_OK;
}


namespace mx { namespace io {

#define MXSIMULATORIOTOEASY(fe, key, member) \
    fe = new MxIOElement(); \
    if(toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define MXSIMULATORIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;

template <>
HRESULT toFile(const MxSimulator &dataElement, const MxMetaData &metaData, MxIOElement *fileElement) {

    MxIOElement *fe;

    MXSIMULATORIOTOEASY(fe, "origin", MxVector3d::from(_Engine.s.origin));
    MXSIMULATORIOTOEASY(fe, "dim", MxVector3d::from(_Engine.s.dim));
    MXSIMULATORIOTOEASY(fe, "cutoff", _Engine.s.cutoff);
    MXSIMULATORIOTOEASY(fe, "cells", MxVector3i::from(_Engine.s.cdim));
    MXSIMULATORIOTOEASY(fe, "integrator", (int)_Engine.integrator);
    MXSIMULATORIOTOEASY(fe, "dt", _Engine.dt);
    MXSIMULATORIOTOEASY(fe, "time", _Engine.time);
    MXSIMULATORIOTOEASY(fe, "boundary_conditions", _Engine.boundary_conditions);
    MXSIMULATORIOTOEASY(fe, "max_distance", _Engine.particle_max_dist_fraction * _Engine.s.h[0]);
    MXSIMULATORIOTOEASY(fe, "seed", MxGetSeed());
    
    if(dataElement.app != NULL) {
        auto renderer = dataElement.app->getRenderer();

        if(renderer != NULL && renderer->clipPlaneCount() > 0) {

            MxVector4f clipPlaneEq;
            MxVector3f normal, point;
            std::vector<MxVector3f> normals, points;

            for(unsigned int i = 0; i < renderer->clipPlaneCount(); i++) {
                std::tie(normal, point) = MxPlaneEquation(renderer->getClipPlaneEquation(i));
                normals.push_back(normal);
                points.push_back(point);
            }

            MXSIMULATORIOTOEASY(fe, "clipPlaneNormals", normals);
            MXSIMULATORIOTOEASY(fe, "clipPlanePoints", points);

        }

    }

    fileElement->type = "Simulator";

    return S_OK;
}

template <>
HRESULT fromFile(const MxIOElement &fileElement, const MxMetaData &metaData, MxSimulator_Config *dataElement) { 

    MxIOChildMap::const_iterator feItr;

    // Do sim setup
    
    MxVector3d origin(0.);
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "origin", &origin);
    dataElement->universeConfig.origin = MxVector3f(origin);

    MxVector3d dim(0.);
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "dim", &dim);
    dataElement->universeConfig.dim = MxVector3f(dim);

    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "cutoff", &dataElement->universeConfig.cutoff);

    MxVector3i cells(0);
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "cells", &cells);
    dataElement->universeConfig.spaceGridSize = cells;
    
    int integrator;
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "integrator", &integrator); 
    dataElement->universeConfig.integrator = (EngineIntegrator)integrator;
    
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "dt", &dataElement->universeConfig.dt);
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "time", &dataElement->universeConfig.start_step);
    
    MxBoundaryConditionsArgsContainer *bcArgs = new MxBoundaryConditionsArgsContainer();
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "boundary_conditions", bcArgs);
    dataElement->universeConfig.setBoundaryConditions(bcArgs);
    
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "max_distance", &dataElement->universeConfig.max_distance);

    unsigned int seed;
    MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "seed", &seed);
    dataElement->setSeed(seed);
    
    if(fileElement.children.find("clipPlaneNormals") != fileElement.children.end()) {
        std::vector<MxVector3f> normals, points;
        std::vector<std::tuple<MxVector3f, MxVector3f> > clipPlanes;
        MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "clipPlaneNormals", &normals);
        MXSIMULATORIOFROMEASY(feItr, fileElement.children, metaData, "clipPlanePoints", &points);
        if(normals.size() > 0) {
            for(unsigned int i = 0; i < normals.size(); i++) 
                clipPlanes.push_back(std::make_tuple(normals[i], points[i]));
            dataElement->clipPlanes = MxParsePlaneEquation(clipPlanes);
        }
    }

    return S_OK;
}

}};
