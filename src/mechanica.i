%module mechanica

%include "typemaps.i"

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"
%include "std_unordered_map.i"

// C++ std::set handling
%include "std_set.i"

// C++ std::vector handling
%include "std_vector.i"

// C++ std::list handling
%include "std_list.i"

// C++ std::pair handling
%include "std_pair.i"

%include "stl.i"

// STL exception handling
%include "exception.i"

%include "cpointer.i"

#define CAPI_DATA(RTYPE) RTYPE
#define CAPI_FUNC(RTYPE) RTYPE
#define CAPI_EXPORT
#define CAPI_STRUCT(RTYPE) struct RTYPE

// Lie to SWIG; so long as these aren't passed to the C compiler, no problem
#define __attribute__(x)
#define aligned(x)

%{

    #define SWIG_FILE_WITH_INIT

    #include "mechanica_private.h"

    // todo: A little hacking here; implement a more sustainable cross-platform solution
    #ifndef M_PI
        #define M_PI       3.14159265358979323846   // pi
    #endif

%}

// config stuff
%ignore MX_MODEL_DIR;
%include "mx_config.h"

%pythoncode %{
    __version__ = str(MX_VERSION) + '-dev' + str(MX_VERSION_DEV)
    MX_BUILD_DATE = _mechanica.mxBuildDate()
    MX_BUILD_TIME = _mechanica.mxBuildTime()

    class version:
        version = __version__
        system_name = MX_SYSTEM_NAME
        system_version = MX_SYSTEM_VERSION
        compiler = MX_COMPILER_ID
        compiler_version = MX_COMPILER_VERSION
        build_date = MX_BUILD_DATE + ', ' + MX_BUILD_TIME
        major = MX_VERSION_MAJOR
        minor = MX_VERSION_MINOR
        patch = MX_VERSION_PATCH
        dev = MX_VERSION_DEV

        @staticmethod
        def cpuinfo():
            return {k: v for k, v in getFeaturesMap().items()}

        @staticmethod
        def compile_flags():
            cf = MxCompileFlags()
            return {k: cf.getFlag(k) for k in cf.getFlags()}
%}

//                                      Imports

// Logger
%include "MxLogger.i"

// submodule: types
%include "types/types.i"

// submodule: state
%include "state/state.i"

// submodule: event
%include "event/event.i"

// submodule: mdcore
%include "mdcore/include/mdcore.i"

// submodule: rendering
%include "rendering/rendering.i"

// MxUtil
%include "MxUtil.i"

// System
%include "MxSystem.i"

// Simulator
%include "MxSimulator.i"

// SurfaceSimulator
%include "MxSurfaceSimulator.i"

// Universe
%include "MxUniverse.i"


//                                      Post-imports

%pythoncode %{
    
# From MxSimulator

    def close():
        return MxSimulatorPy.get().close()

    def show():
        return MxSimulatorPy.get().show()

    def irun():
        return MxSimulatorPy.get().irun()

    def init(*args, **kwargs):
        return MxSimulatorPy_init(args, kwargs)

    def run(*args, **kwargs):
        return MxSimulatorPy.get()._run(args, kwargs)
    
    # From MxUniverse
    step = MxUniverse.step
    stop = MxUniverse.stop
    start = MxUniverse.start

    # From MxUtils
    test = _MxTest
    random_points = MxRandomPoints
    points = MxPoints
    primes = MxMath_FindPrimes
    cpuinfo = MxSystem.cpu_info()
    compile_flags = MxSystem.compile_flags()
%}
