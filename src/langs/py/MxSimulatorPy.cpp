/**
 * @file MxSimulatorPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxSimulator
 * @date 2022-03-23
 * 
 */

#include "MxSimulatorPy.h"

#include "MxBoundaryConditionsPy.h"
#include "MxSystemPy.h"

#include <MxLogger.h>
#include <MxUtil.h>
#include <rendering/MxApplication.h>
#include <rendering/MxClipPlane.hpp>
#include <rendering/MxGlfwApplication.h>
#include <rendering/MxWindowlessApplication.h>


#define SIMPY_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define SIMULATORPY_CHECK()  if (!MxSimulator::get()) { return mx_error(E_INVALIDARG, "Simulator is not initialized"); }

#define SIMPY_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = MX_FUNCTION; \
            err += "universe not initialized"; \
            mx_exp(std::domain_error(err.c_str())); \
        }

#define SIMPY_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        mx_exp(e); return retval; \
    }

/**
 * Make a Arguments struct from a string list,
 * Magnum has different args for different app types,
 * so this needs to be a template.
 */
template<typename T>
struct ArgumentsWrapper  {

    ArgumentsWrapper(PyObject *args) {

        for(int i = 0; i < PyList_Size(args); ++i) {
            PyObject *o = PyList_GetItem(args, i);
            strings.push_back(mx::cast<PyObject, std::string>(o));
            cstrings.push_back(strings.back().c_str());

            Log(LOG_INFORMATION) <<  "args: " << cstrings.back() ;;
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

static void parse_kwargs(PyObject *kwargs, MxSimulator_Config &conf) {

    Log(LOG_INFORMATION) << "parsing python dictionary input";

    PyObject *o;

    std::string loadPath;
    if((o = PyDict_GetItemString(kwargs, "load_file"))) {
        loadPath = mx::cast<PyObject, std::string>(o);

        Log(LOG_INFORMATION) << "got load file: " << loadPath;
        
        if(initSimConfigFromFile(loadPath, conf) != S_OK) 
            return;
    }

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
        cells = new MxVector3i(mx::cast<PyObject, MxVector3i>(o));

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

    MxBoundaryConditionsArgsContainerPy *bcArgs;
    if((o = PyDict_GetItemString(kwargs, "bc"))) {
        bcArgs = new MxBoundaryConditionsArgsContainerPy(o);
        
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

        Log(LOG_INFORMATION) << "got windowless " << (*windowless ? "True" : "False");
    }
    else windowless = NULL;

    MxVector2i *window_size;
    if((o = PyDict_GetItemString(kwargs, "window_size"))) {
        window_size = new MxVector2i(mx::cast<PyObject, Magnum::Vector2i>(o));

        Log(LOG_INFORMATION) << "got window_size: " << std::to_string(window_size->x()) << "," << std::to_string(window_size->y());
    }
    else window_size = NULL;

    unsigned int *seed;
    if((o = PyDict_GetItemString(kwargs, "seed"))) {
        seed = new unsigned int(mx::cast<PyObject, unsigned int>(o));

        Log(LOG_INFORMATION) << "got seed: " << std::to_string(*seed);
    } 
    else seed = NULL;

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

    if(dim)

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

    if(bcArgs) conf.universeConfig.setBoundaryConditions((MxBoundaryConditionsArgsContainer*)bcArgs);
    
    if(max_distance) conf.universeConfig.max_distance = *max_distance;
    if(windowless) conf.setWindowless(*windowless);
    if(window_size) conf.setWindowSize(*window_size);
    if(seed) conf.setSeed(*seed);
    if(perfcounters) conf.universeConfig.timers_mask = *perfcounters;
    if(perfcounter_period) conf.universeConfig.timer_output_period = *perfcounter_period;
    if(logger_level) MxLogger::setLevel(*logger_level);
    if(clip_planes) conf.clipPlanes = MxParsePlaneEquation(*clip_planes);
}

static PyObject *ih = NULL;

HRESULT _setIPythonInputHook(PyObject *_ih) {
    ih = _ih;
    return S_OK;
}

HRESULT _onIPythonNotReady() {
    MxSimulator::get()->app->mainLoopIteration(0.001);
    return S_OK;
}

static void simulator_interactive_run() {
    Log(LOG_INFORMATION) <<  "entering ";

    if (MxUniverse_Flag(MxUniverse_Flags::MX_POLLING_MSGLOOP)) {
        return;
    }

    // interactive run only works in terminal ipytythn.
    PyObject *ipy = MxIPython_Get();
    const char* ipyname = ipy ? ipy->ob_type->tp_name : "NULL";
    Log(LOG_INFORMATION) <<  "ipy type: " << ipyname;

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
        
        Log(LOG_INFORMATION) <<  "pt_inputhooks: " << mx::str(pt_inputhooks);
        
        PyObject *reg = PyObject_GetAttrString(pt_inputhooks, "register");
        
        Log(LOG_INFORMATION) <<  "reg: " << mx::str(reg);
        
        Log(LOG_INFORMATION) <<  "ih: " << mx::str(ih);
        
        Log(LOG_INFORMATION) <<  "calling reg....";
        
        PyObject *args = PyTuple_Pack(2, mx_str, ih);
        PyObject *reg_result = PyObject_Call(reg, args, NULL);
        Py_XDECREF(args);
        
        if(reg_result == NULL) {
            mx_exp(std::logic_error("error calling IPython.terminal.pt_inputhooks.register()"));
        }
        
        Py_XDECREF(reg_result);

        // import IPython
        // ip = IPython.get_ipython()
        PyObject *ipython = PyImport_ImportString("IPython");
        Log(LOG_INFORMATION) <<  "ipython: " << mx::str(ipython);
        
        PyObject *get_ipython = PyObject_GetAttrString(ipython, "get_ipython");
        Log(LOG_INFORMATION) <<  "get_ipython: " << mx::str(get_ipython);
        
        args = PyTuple_New(0);
        PyObject *ip = PyObject_Call(get_ipython, args, NULL);
        Py_XDECREF(args);
        
        if(ip == NULL) {
            mx_exp(std::logic_error("error calling IPython.get_ipython()"));
        }
        
        PyObject *enable_gui = PyObject_GetAttrString(ip, "enable_gui");
        
        if(enable_gui == NULL) {
            mx_exp(std::logic_error("error calling ipython has no enable_gui attribute"));
        }
        
        args = PyTuple_Pack(1, mx_str);
        PyObject *enable_gui_result = PyObject_Call(enable_gui, args, NULL);
        Py_XDECREF(args);
        Py_XDECREF(mx_str);
        
        if(enable_gui_result == NULL) {
            mx_exp(std::logic_error("error calling ipython.enable_gui(\"mechanica\")"));
        }
        
        Py_XDECREF(enable_gui_result);

        MxUniverse_SetFlag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP, true);

        // show the app
        MxSimulator::get()->app->show();
    }
    else {
        // not in ipython, so run regular run.
        MxSimulator::get()->run(-1);
        return;
    }

    Py_XDECREF(ipy);
    Log(LOG_INFORMATION) << "leaving ";
}

MxSimulatorPy *MxSimulatorPy::get() {
    return (MxSimulatorPy*)MxSimulator::get();
}

PyObject *MxSimulatorPy_init(PyObject *args, PyObject *kwargs) {

    std::thread::id id = std::this_thread::get_id();
    Log(LOG_INFORMATION) << "thread id: " << id;

    try {

        if(MxSimulator::get()) {
            mx_exp(std::domain_error( "Error, Simulator is already initialized" ));
        }
        
        MxSimulator *sim = new MxSimulator();

        #ifdef MX_WITHCUDA
        MxCUDA::init();
        MxCUDA::setGLDevice(0);
        sim->makeCUDAConfigCurrent(new MxSimulatorCUDAConfig());
        #endif

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
                mx_exp(std::logic_error("could not get argv from sys module"));
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
        if(MxPy_ZMQInteractiveShell()) {
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
        
        if(!conf.windowless() && MxPy_ZMQInteractiveShell()) {
            Log(LOG_WARNING) << "requested window mode in Jupyter notebook, will fail badly if there is no X-server";
        }

        MxSetSeed(const_cast<MxSimulator_Config&>(conf).seed());

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

                mx_exp(std::domain_error("could not create windowless gl context"));
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
                mx_exp(std::domain_error("could not create  gl context"));
            }
            else {
                sim->app = glfwApp;
            }
        }

        Log(LOG_INFORMATION) << "sucessfully created application";

        sim->makeCurrent();
        
        if(MxPy_ZMQInteractiveShell()) {
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
    
    SIMULATORPY_CHECK();

    MxUniverse_SetFlag(MX_RUNNING, true);

    Log(LOG_DEBUG) << "checking for ipython";

    bool interactive = MxPy_TerminalInteractiveShell();
    Mx_setTerminalInteractiveShell(interactive);

    if (interactive) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        Log(LOG_DEBUG) <<  "in ipython, calling interactive";

        MxSimulator::get()->app->show();
        
        Log(LOG_DEBUG) << "finished";

        return S_OK;
    }
    else {
        Log(LOG_DEBUG) << "not ipython, returning MxSimulator_Run";
        return MxSimulator::get()->run(-1);
    }
}

HRESULT MxSimulatorPy::_show()
{
    SIMULATORPY_CHECK();
    
    Log(LOG_DEBUG) << "checking for ipython";

    bool interactive = MxPy_TerminalInteractiveShell();
    Mx_setTerminalInteractiveShell(interactive);
    
    if (interactive) {

        if (!MxUniverse_Flag(MxUniverse_Flags::MX_IPYTHON_MSGLOOP)) {
            // ipython message loop, this exits right away
            simulator_interactive_run();
        }

        Log(LOG_TRACE) << "in ipython, calling interactive";

        MxSimulator::get()->app->show();
        
        Log(LOG_INFORMATION) << ", MxSimulator::get()->app->show() all done" ;

        return S_OK;
    }
    else {
        Log(LOG_TRACE) << "not ipython, returning MxSimulator::get()->app->show()";
        return MxSimulator::get()->show();
    }
}

void *MxSimulatorPy::wait_events(const double &timeout) {
    SIMPY_TRY();
    if(timeout < 0) {
        SIMPY_CHECK(MxSimulator::waitEvents());
    }
    else {
        SIMPY_CHECK(MxSimulator::waitEventsTimeout(timeout));
    }
    SIMPY_FINALLY(NULL);
}

PyObject *MxSimulatorPy::_run(PyObject *args, PyObject *kwargs) {
    SIMPY_TRY();

    if (MxPy_ZMQInteractiveShell()) {
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
    SIMPY_CHECK(MxSimulator::get()->run(et));
    SIMPY_FINALLY(NULL);
}
