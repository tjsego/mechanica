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
#include <MxPy.h>

// mdcore errs.h
#include <errs.h>

#include <thread>

static std::vector<MxVector3f> fillCubeRandom(const MxVector3f &corner1, const MxVector3f &corner2, int nParticles);

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
            throw std::domain_error(err.c_str()); \
        }

#define SIM_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define SIM_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        mx_exp(e); return retval; \
    }

static MxSimulator* Simulator = NULL;

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

#define PY_CHECK(hr) {if(!SUCCEEDED(hr)) { throw py::error_already_set();}}

#define PYSIMULATOR_CHECK() { \
    if(!Simulator) { \
        throw std::domain_error(std::string("Simulator Error in ") + MX_FUNCTION + ": Simulator not initialized"); \
    } \
}

/**
 * Make a Arguments struct from a python string list,
 * Agh!!! Magnum has different args for different app types,
 * so this needs to be a damned template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(const std::vector<std::string> &args) {

        for(auto &a : args) {
            strings.push_back(a);
            cstrings.push_back(a.c_str());

            Log(LOG_INFORMATION) <<  "args: " << a ;;
        }

        // stupid thing is a int reference, keep an ivar around for it
        // to point to.
        argsSeriouslyTakesAFuckingIntReference = cstrings.size();
        char** fuckingConstBullshit = const_cast<char**>(cstrings.data());

        pArgs = new T(argsSeriouslyTakesAFuckingIntReference, fuckingConstBullshit);
    }

    ArgumentsWrapper(PyObject *args) {

        for(int i = 0; i < PyList_Size(args); ++i) {
            PyObject *o = PyList_GetItem(args, i);
            strings.push_back(mx::cast<PyObject, std::string>(o));
            cstrings.push_back(strings.back().c_str());

            Log(LOG_INFORMATION) <<  "args: " << cstrings.back() ;;
        }

        // stupid thing is a int reference, keep an ivar around for it
        // to point to.
        argsSeriouslyTakesAFuckingIntReference = cstrings.size();
        char** fuckingConstBullshit = const_cast<char**>(cstrings.data());

        pArgs = new T(argsSeriouslyTakesAFuckingIntReference, fuckingConstBullshit);
    }

    ~ArgumentsWrapper() {
        delete pArgs;
    }


    // OMG this is a horrible design.
    // how I hate C++
    std::vector<std::string> strings;
    std::vector<const char*> cstrings;
    T *pArgs = NULL;
    int argsSeriouslyTakesAFuckingIntReference;
};

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
                throw std::logic_error(msg);
            }
        }
    }
    if(dt) conf.universeConfig.dt = *dt;

    if(bcArgs) conf.universeConfig.setBoundaryConditions(bcArgs);
    else conf.universeConfig.setBoundaryConditions(new MxBoundaryConditionsArgsContainer(bcValue, bcVals, bcVels, bcRestores));
    
    if(max_distance) conf.universeConfig.max_distance = *max_distance;
    if(windowless) conf.setWindowless(*windowless);
    if(window_size) conf.setWindowSize(*window_size);
    if(perfcounters) conf.universeConfig.timers_mask = *perfcounters;
    if(perfcounter_period) conf.universeConfig.timer_output_period = *perfcounter_period;
    if(logger_level) MxLogger::setLevel(*logger_level);
    if(clip_planes) conf.clipPlanes = MxParsePlaneEquation(*clip_planes);
}

// intermediate kwarg parsing
static void parse_kwargs(const std::vector<std::string> &kwargs, MxSimulator_Config &conf) {

    Log(LOG_INFORMATION) << "parsing vector string input";

    std::string s;

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

        Log(LOG_INFORMATION) << "got windowless" << *windowless ? "True" : "False";
    }
    else windowless = NULL;

    MxVector2i *window_size;
    if(mx::parse::has_kwarg(kwargs, "window_size")) {
        s = mx::parse::kwargVal(kwargs, "window_size");
        window_size = new MxVector2i(mx::parse::strToVec<int>(s));

        Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

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
                 perfcounters, 
                 perfcounter_period, 
                 logger_level, 
                 clip_planes);
}

// python support: intermediate kwarg parsing
// todo: consolidate MxSimulatorPy and eliminate this entirely
static void parse_kwargs(PyObject *kwargs, MxSimulator_Config &conf) {

    Log(LOG_INFORMATION) << "parsing python dictionary input";

    PyObject *o;

    MxVector3f *dim;
    if((o = PyDict_GetItemString(kwargs, "dim"))) {
        dim = new MxVector3f(mx::cast<PyObject, Magnum::Vector3>(o));

        Log(LOG_INFORMATION) << "got dim: " 
                             << std::to_string(dim->x()) << "," 
                             << std::to_string(dim->y()) << "," 
                             << std::to_string(dim->z());
    }
    else dim = NULL;

    double *cutoff;
    if((o = PyDict_GetItemString(kwargs, "cutoff"))) {
        cutoff = new double(mx::cast<PyObject, double>(o));

        Log(LOG_INFORMATION) << "got cutoff: " << std::to_string(*cutoff);
    }
    else cutoff = NULL;

    MxVector3i *cells;
    if((o = PyDict_GetItemString(kwargs, "cells"))) {
        cells = new MxVector3i(mx::cast<PyObject, Vector3i>(o));

        Log(LOG_INFORMATION) << "got cells: " 
                             << std::to_string(cells->x()) << "," 
                             << std::to_string(cells->y()) << "," 
                             << std::to_string(cells->z());
    }
    else cells = NULL;

    unsigned *threads;
    if((o = PyDict_GetItemString(kwargs, "threads"))) {
        threads = new unsigned(mx::cast<PyObject, unsigned>(o));

        Log(LOG_INFORMATION) << "got threads: " << std::to_string(*threads);
    }
    else threads = NULL;

    int *integrator;
    if((o = PyDict_GetItemString(kwargs, "integrator"))) {
        integrator = new int(mx::cast<PyObject, int>(o));

        Log(LOG_INFORMATION) << "got integrator: " << std::to_string(*integrator);
    }
    else integrator = NULL;

    double *dt;
    if((o = PyDict_GetItemString(kwargs, "dt"))) {
        dt = new double(mx::cast<PyObject, double>(o));

        Log(LOG_INFORMATION) << "got dt: " << std::to_string(*dt);
    }
    else dt = NULL;

    MxBoundaryConditionsArgsContainer *bcArgs;
    if((o = PyDict_GetItemString(kwargs, "bc"))) {
        bcArgs = new MxBoundaryConditionsArgsContainer(o);
        
        Log(LOG_INFORMATION) << "Got boundary conditions";
    }
    else bcArgs = NULL;

    double *max_distance;
    if((o = PyDict_GetItemString(kwargs, "max_distance"))) {
        max_distance = new double(mx::cast<PyObject, double>(o));

        Log(LOG_INFORMATION) << "got max_distance: " << std::to_string(*max_distance);
    }
    else max_distance = NULL;

    bool *windowless;
    if((o = PyDict_GetItemString(kwargs, "windowless"))) {
        windowless = new bool(mx::cast<PyObject, bool>(o));

        Log(LOG_INFORMATION) << "got windowless" << *windowless ? "True" : "False";
    }
    else windowless = NULL;

    MxVector2i *window_size;
    if((o = PyDict_GetItemString(kwargs, "window_size"))) {
        window_size = new MxVector2i(mx::cast<PyObject, Magnum::Vector2i>(o));

        Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

    uint32_t *perfcounters;
    if((o = PyDict_GetItemString(kwargs, "perfcounters"))) {
        perfcounters = new uint32_t(mx::cast<PyObject, uint32_t>(o));

        Log(LOG_INFORMATION) << "got perfcounters: " << std::to_string(*perfcounters);
    }
    else perfcounters = NULL;

    int *perfcounter_period;
    if((o = PyDict_GetItemString(kwargs, "perfcounter_period"))) {
        perfcounter_period = new int(mx::cast<PyObject, int>(o));

        Log(LOG_INFORMATION) << "got perfcounter_period: " << std::to_string(*perfcounter_period);
    }
    else perfcounter_period = NULL;
    
    int *logger_level;
    if((o = PyDict_GetItemString(kwargs, "logger_level"))) {
        logger_level = new int(mx::cast<PyObject, int>(o));

        Log(LOG_INFORMATION) << "got logger_level: " << std::to_string(*logger_level);
    }
    else logger_level = NULL;
    
    std::vector<std::tuple<MxVector3f, MxVector3f> > *clip_planes;
    PyObject *pyTuple;
    PyObject *pyTupleItem;
    if((o = PyDict_GetItemString(kwargs, "clip_planes"))) {
        clip_planes = new std::vector<std::tuple<MxVector3f, MxVector3f> >();
        for(unsigned int i=0; i < PyList_Size(o); ++i) {
            pyTuple = PyList_GetItem(o, i);
            if(!pyTuple || !PyTuple_Check(pyTuple)) {
                Log(LOG_ERROR) << "Clip plane input entry not a tuple";
                continue;
            }
            
            pyTupleItem = PyTuple_GetItem(pyTuple, 0);
            if(!PyList_Check(pyTupleItem)) {
                Log(LOG_ERROR) << "Clip plane point entry not a list";
                continue;
            }
            MxVector3f point = mx::cast<PyObject, MxVector3f>(pyTupleItem);

            pyTupleItem = PyTuple_GetItem(pyTuple, 1);
            if(!PyList_Check(pyTupleItem)) {
                Log(LOG_ERROR) << "Clip plane normal entry not a list";
                continue;
            }
            MxVector3f normal = mx::cast<PyObject, MxVector3f>(pyTupleItem);

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

static PyObject *simulator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return NULL;
}

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

HRESULT modules_init() {
    Log(LOG_DEBUG) << ", initializing modules... " ;

    _MxParticle_init();
    _MxCluster_init();

    return S_OK;
}

int universe_init(const MxUniverseConfig &conf ) {

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
        throw std::runtime_error(errs_getstring(0));
    }

    if ( modules_init() != S_OK ) {
        Log(LOG_ERROR) << errs_getstring(0);
        throw std::runtime_error(errs_getstring(0));
    }

    fflush(stdout);

    return 0;
}

static std::vector<MxVector3f> fillCubeRandom(const MxVector3f &corner1, const MxVector3f &corner2, int nParticles) {
    std::vector<MxVector3f> result;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> disx(corner1[0], corner2[0]);
    std::uniform_real_distribution<float> disy(corner1[1], corner2[1]);
    std::uniform_real_distribution<float> disz(corner1[2], corner2[2]);

    for(int i = 0; i < nParticles; ++i) {
        result.push_back(MxVector3f{disx(gen), disy(gen), disz(gen)});

    }

    return result;
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
            throw std::domain_error("Error, Simulator is already initialized" );
        }
        
        MxSimulator *sim = new MxSimulator();
        
        Universe.name = conf.title();

        Log(LOG_INFORMATION) << "got universe name: " << Universe.name;

        // init the engine first
        /* Initialize scene particles */
        universe_init(conf.universeConfig);

        if(conf.windowless()) {
            Log(LOG_INFORMATION) <<  "creating Windowless app" ;
            
            ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(appArgv);

            MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

            if(FAILED(windowlessApp->createContext(conf))) {
                delete windowlessApp;

                throw std::domain_error("could not create windowless gl context");
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
                throw std::domain_error("could not create  gl context");
            }
            else {
                sim->app = glfwApp;
            }
        }

        Log(LOG_INFORMATION) << "sucessfully created application";

        Simulator = sim;
        
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

static void simulator_interactive_run() {
    Log(LOG_INFORMATION) <<  "entering ";

    if (MxUniverse_Flag(MxUniverse_Flags::MX_POLLING_MSGLOOP)) {
        return;
    }

    // interactive run only works in terminal ipytythn.
    PyObject *ipy = MxIPython_Get();
    const char* ipyname = ipy ? ipy->ob_type->tp_name : "NULL";
    Log(LOG_INFORMATION) <<  "ipy type: " << ipyname ;;

    if(ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {

        Log(LOG_DEBUG) << "calling python interactive loop";
        
        PyObject *mx_str = mx::cast<std::string, PyObject*>(std::string("mechanica"));

        // Try to import ipython

        /**
         *        """
            Registers the mechanica input hook with the ipython pt_inputhooks
            class.

            The ipython TerminalInteractiveShell.enable_gui('name') method
            looks in the registered input hooks in pt_inputhooks, and if it
            finds one, it activtes that hook.

            To acrtivate the gui mode, call:

            ip = IPython.get_ipython()
            ip.
            """
            import IPython.terminal.pt_inputhooks as pt_inputhooks
            pt_inputhooks.register("mechanica", inputhook)
         *
         */

        PyObject *pt_inputhooks = PyImport_ImportString("IPython.terminal.pt_inputhooks");
        
        Log(LOG_INFORMATION) <<  "pt_inputhooks: " << mx::cast<PyObject, std::string>(pt_inputhooks) ;;
        
        PyObject *reg = PyObject_GetAttrString(pt_inputhooks, "register");
        
        Log(LOG_INFORMATION) <<  "reg: " << mx::cast<PyObject, std::string>(reg) ;;
        
        PyObject *ih = (PyObject*)&MxSimulatorPy::_input_hook;
        
        Log(LOG_INFORMATION) <<  "ih: " << mx::cast<PyObject, std::string>(ih) ;;

        //py::cpp_function ih(MxSimulatorPy::_input_hook);
        
        //reg("mechanica", ih);
        
        Log(LOG_INFORMATION) <<  "calling reg...." ;;
        
        PyObject *args = PyTuple_Pack(2, mx_str, ih);
        PyObject *reg_result = PyObject_Call(reg, args, NULL);
        Py_XDECREF(args);
        
        if(reg_result == NULL) {
            throw std::logic_error("error calling IPython.terminal.pt_inputhooks.register()");
        }
        
        Py_XDECREF(reg_result);

        // import IPython
        // ip = IPython.get_ipython()
        PyObject *ipython = PyImport_ImportString("IPython");
        Log(LOG_INFORMATION) <<  "ipython: " << mx::cast<PyObject, std::string>(ipython) ;;
        
        PyObject *get_ipython = PyObject_GetAttrString(ipython, "get_ipython");
        Log(LOG_INFORMATION) <<  "get_ipython: " << mx::cast<PyObject, std::string>(get_ipython) ;;
        
        args = PyTuple_New(0);
        PyObject *ip = PyObject_Call(get_ipython, args, NULL);
        Py_XDECREF(args);
        
        if(ip == NULL) {
            throw std::logic_error("error calling IPython.get_ipython()");
        }
        
        PyObject *enable_gui = PyObject_GetAttrString(ip, "enable_gui");
        
        if(enable_gui == NULL) {
            throw std::logic_error("error calling ipython has no enable_gui attribute");
        }
        
        args = PyTuple_Pack(1, mx_str);
        PyObject *enable_gui_result = PyObject_Call(enable_gui, args, NULL);
        Py_XDECREF(args);
        Py_XDECREF(mx_str);
        
        if(enable_gui_result == NULL) {
            throw std::logic_error("error calling ipython.enable_gui(\"mechanica\")");
        }
        
        Py_XDECREF(enable_gui_result);

        MxUniverse_SetFlag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP, true);

        // show the app
        Simulator->app->show();
    }
    else {
        // not in ipython, so run regular run.
        Simulator->run(-1);
        return;
    }

    Py_XDECREF(ipy);
    Log(LOG_INFORMATION) << "leaving ";
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

    // init the engine first
    /* Initialize scene particles */
    universe_init(conf.universeConfig);


    if(conf.windowless()) {

        /*



        MxWindowlessApplication::Configuration windowlessConf;

        MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

        if(!windowlessApp->tryCreateContext(conf)) {
            delete windowlessApp;

            throw std::domain_error("could not create windowless gl context");
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

    Simulator = sim;

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
 * gets the global simulator object, throws exception if fail.
 */
MxSimulator *MxSimulator::get() {
    if(Simulator) {
        return Simulator;
    }
    throw std::logic_error("Simulator is not initiazed");
}

MxSimulatorPy *MxSimulatorPy::get() {
    return (MxSimulatorPy*)MxSimulator::get();
}

PyObject *MxSimulatorPy_init(PyObject *args, PyObject *kwargs) {

    std::thread::id id = std::this_thread::get_id();
    Log(LOG_INFORMATION) << "thread id: " << id;

    try {

        if(Simulator) {
            throw std::domain_error( "Error, Simulator is already initialized" );
        }
        
        MxSimulator *sim = new MxSimulator();

        Log(LOG_INFORMATION) << "successfully created new simulator";

        // get the argv,
        PyObject * argv = NULL;
        if(kwargs == NULL || (argv = PyDict_GetItemString(kwargs, "argv")) == NULL) {
            Log(LOG_INFORMATION) << "Getting command-line args";

            PyObject *sys_name = mx::cast<std::string, PyObject*>(std::string("sys"));
            PyObject *sys = PyImport_Import(sys_name);
            argv = PyObject_GetAttrString(sys, "argv");
            
            Py_DECREF(sys_name);
            Py_DECREF(sys);
            
            if(!argv) {
                throw std::logic_error("could not get argv from sys module");
            }
        }

        MxSimulator_Config conf;
        
        if(PyList_Size(argv) > 0) {
            std::string name = mx::cast<PyObject, std::string>(PyList_GetItem(argv, 0));
            Universe.name = name;
            conf.setTitle(name);
        }

        Log(LOG_INFORMATION) << "got universe name: " << Universe.name;
        
        // find out if we are in jupyter, set default state of config,
        // not sure if this makes more sense in config constructor or here...
        if(Mx_ZMQInteractiveShell()) {
            Log(LOG_INFORMATION) << "in zmq shell, setting windowless default to true";
            conf.setWindowless(true);
        }
        else {
            Log(LOG_INFORMATION) << "not zmq shell, setting windowless default to false";
            conf.setWindowless(false);
        }

        if(kwargs && PyDict_Size(kwargs) > 0) {
            parse_kwargs(kwargs, conf);
        }

        Log(LOG_INFORMATION) << "successfully parsed args";
        
        if(!conf.windowless() && Mx_ZMQInteractiveShell()) {
            Log(LOG_WARNING) << "requested window mode in Jupyter notebook, will fail badly if there is no X-server";
        }

        // init the engine first
        /* Initialize scene particles */
        universe_init(conf.universeConfig);

        Log(LOG_INFORMATION) << "successfully initialized universe";

        if(conf.windowless()) {
            Log(LOG_INFORMATION) <<  "creating Windowless app" ;
            
            ArgumentsWrapper<MxWindowlessApplication::Arguments> margs(argv);

            MxWindowlessApplication *windowlessApp = new MxWindowlessApplication(*margs.pArgs);

            if(FAILED(windowlessApp->createContext(conf))) {
                delete windowlessApp;

                throw std::domain_error("could not create windowless gl context");
            }
            else {
                sim->app = windowlessApp;
            }

	    Log(LOG_TRACE) << "sucessfully created windowless app";
        }
        else {
            Log(LOG_INFORMATION) <<  "creating GLFW app" ;
            
            ArgumentsWrapper<MxGlfwApplication::Arguments> margs(argv);

            MxGlfwApplication *glfwApp = new MxGlfwApplication(*margs.pArgs);
            
            if(FAILED(glfwApp->createContext(conf))) {
                Log(LOG_DEBUG) << "deleting failed glfwApp";
                delete glfwApp;
                throw std::domain_error("could not create  gl context");
            }
            else {
                sim->app = glfwApp;
            }
        }

        Log(LOG_INFORMATION) << "sucessfully created application";

        Simulator = sim;
        
        if(Mx_ZMQInteractiveShell()) {
            Log(LOG_INFORMATION) << "in jupyter notebook, calling widget init";
            PyObject *widgetInit = MxSystemPy::jwidget_init(args, kwargs);
            if(!widgetInit) {
                Log(LOG_ERROR) << "could not create jupyter widget";
                return NULL;
            }
            else {
                Py_DECREF(widgetInit);
            }
        }
        
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        Log(LOG_CRITICAL) << "Initializing simulator failed!";

        mx_exp(e); return NULL;
    }
}

HRESULT MxSimulatorPy::irun()
{
    Log(LOG_TRACE);
    
    SIMULATOR_CHECK();

    MxUniverse_SetFlag(MX_RUNNING, true);


    Log(LOG_DEBUG) << "checking for ipython";
    if (Mx_TerminalInteractiveShell()) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        Log(LOG_DEBUG) <<  "in ipython, calling interactive";

        Simulator->app->show();
        
        Log(LOG_DEBUG) << "finished";

        return S_OK;
    }
    else {
        Log(LOG_DEBUG) << "not ipython, returning MxSimulator_Run";
        return Simulator->run(-1);
    }
}

HRESULT MxSimulatorPy::_show()
{
    SIMULATOR_CHECK();

    Log(LOG_TRACE) << "checking for ipython";
    
    if (Mx_TerminalInteractiveShell()) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        Log(LOG_TRACE) << "in ipython, calling interactive";

        Simulator->app->show();
        
        Log(LOG_INFORMATION) << ", Simulator->app->show() all done" ;

        return S_OK;
    }
    else {
        Log(LOG_TRACE) << "not ipython, returning Simulator->app->show()";
        return Simulator->show();
    }
}

PyObject *MxSimulatorPy::_input_hook(PyObject *const *args, Py_ssize_t nargs) {
    SIM_TRY();
    
    if(nargs < 1) {
        throw std::logic_error("argument count to mechanica ipython input hook is 0");
    }
    
    PyObject *context = args[0];
    if(context == NULL) {
        throw std::logic_error("mechanica ipython input hook context argument is NULL");
    }
    
    PyObject *input_is_ready = PyObject_GetAttrString(context, "input_is_ready");
    if(input_is_ready == NULL) {
        throw std::logic_error("mechanica ipython input hook context has no \"input_is_ready\" attribute");
    }
    
    PyObject *input_args = PyTuple_New(0);
    
    auto get_ready = [input_is_ready, input_args]() -> bool {
        PyObject *ready = PyObject_Call(input_is_ready, input_args, NULL);
        if(!ready) {
            PyObject* err = PyErr_Occurred();
            std::string str = "error calling input_is_ready";
            str += mx::cast<PyObject, std::string>(err);
            throw std::logic_error(str);
        }
        
        bool bready = mx::cast<PyObject, bool>(ready);
        Py_DECREF(ready);
        return bready;
    };
    
    Py_XDECREF(input_args);
    
    while(!get_ready()) {
        Simulator->app->mainLoopIteration(0.001);
    }
    
    Py_RETURN_NONE;
    
    SIM_FINALLY(NULL);
}

void *MxSimulatorPy::wait_events(const double &timeout) {
    SIM_TRY();
    if(timeout < 0) {
        SIM_CHECK(MxSimulator::waitEvents());
    }
    else {
        SIM_CHECK(MxSimulator::waitEventsTimeout(timeout));
    }
    SIM_FINALLY(NULL);
}

PyObject *MxSimulatorPy::_run(PyObject *args, PyObject *kwargs) {
    SIM_TRY();

    if (Mx_ZMQInteractiveShell()) {
        PyObject* result = MxSystemPy::jwidget_run(args, kwargs);
        if (!result) {
            Log(LOG_ERROR) << "failed to call mechanica.jwidget.run";
            return NULL;
        }

        if (result == Py_True) {
            Py_DECREF(result);
            Py_RETURN_NONE;
        }
        else if (result == Py_False) {
            Log(LOG_INFORMATION) << "returned false from  mechanica.jwidget.run, performing normal simulation";
        }
        else {
            Log(LOG_WARNING) << "unexpected result from mechanica.jwidget.run , performing normal simulation"; 
        }

        Py_DECREF(result);
    }


    double et = mx::arg("et", 0, args, kwargs, -1.0);
    SIM_CHECK(Simulator->run(et));
    SIM_FINALLY(NULL);
}

struct MxUniverseRenderer *MxSimulator::getRenderer() {
    return app->getRenderer();
}
