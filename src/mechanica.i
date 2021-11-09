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

%include "stdint.i"
// STL exception handling
%include "exception.i"

%include "cpointer.i"

#define CAPI_DATA(RTYPE) RTYPE
#define CAPI_FUNC(RTYPE) RTYPE
#define CAPI_EXPORT
#define CAPI_STRUCT(RTYPE) struct RTYPE

// Lie to SWIG; so long as these aren't passed to the C compiler, no problem
#define __attribute__(x)
#define MX_ALIGNED(RTYPE, VAL) RTYPE

%begin %{
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#endif
%}

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
        """Mechanica version

        :meta hide-value:
        """

        system_name = MX_SYSTEM_NAME
        """System name

        :meta hide-value:
        """

        system_version = MX_SYSTEM_VERSION
        """System version

        :meta hide-value:
        """

        compiler = MX_COMPILER_ID
        """Package compiler ID

        :meta hide-value:
        """

        compiler_version = MX_COMPILER_VERSION
        """Package compiler version

        :meta hide-value:
        """

        build_date = MX_BUILD_DATE + ', ' + MX_BUILD_TIME
        """Package build date

        :meta hide-value:
        """

        major = MX_VERSION_MAJOR
        """Mechanica major version

        :meta hide-value:
        """

        minor = MX_VERSION_MINOR
        """Mechanica minor version

        :meta hide-value:
        """

        patch = MX_VERSION_PATCH
        """Mechanica patch version

        :meta hide-value:
        """

        dev = MX_VERSION_DEV
        """Mechanica development stage

        :meta hide-value:
        """

        @staticmethod
        def cpuinfo():
            """Dictionary of CPU info"""
            return {k: v for k, v in getFeaturesMap().items()}

        @staticmethod
        def compile_flags():
            """Dictionary of compiler flags"""
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

// models
%include "models/models.i"

//                                      Post-imports

%pythoncode %{
    
# From MxSimulator

    def close():
        """
        Alias of :meth:`mechanica.mechanica.MxSimulatorPy.close`
        """
        return MxSimulatorPy.get().close()

    def show():
        """
        Alias of :meth:`mechanica.mechanica.MxSimulatorPy.show`
        """
        return MxSimulatorPy.get().show()

    def irun():
        """
        Alias of :meth:`mechanica.mechanica.MxSimulatorPy.irun`
        """
        return MxSimulatorPy.get().irun()

    def init(*args, **kwargs):
        """
        Initialize a simulation in Python

        :type args: PyObject
        :param args: positional arguments; first argument is name of simulation (if any)
        :type kwargs: PyObject
        :param kwargs: keyword arguments; currently supported are

                dim: (3-component list of floats) the dimensions of the spatial domain; default is [10., 10., 10.]

                cutoff: (float) simulation cutoff distance; default is 1.

                cells: (3-component list of ints) the discretization of the spatial domain; default is [4, 4, 4]

                threads: (int) number of threads; default is hardware maximum

                integrator: (int) simulation integrator; default is FORWARD_EULER

                dt: (float) time discretization; default is 0.01

                bc: (int or dict) boundary conditions; default is everywhere periodic

                window_size: (2-component list of ints) size of application window; default is [800, 600]

                logger_level: (int) logger level; default is no logging

                clip_planes: (list of tuple of (MxVector3f, MxVector3f)) list of point-normal pairs of clip planes; default is no planes
        """
        return MxSimulatorPy_init(args, kwargs)

    def run(*args, **kwargs):
        """
        Runs the event loop until all windows close or simulation time expires. 
        Automatically performs universe time propogation. 

        :type args: float
        :param args: final time (default runs infinitly)
        """
        return MxSimulatorPy.get()._run(args, kwargs)
    
    # From MxUniverse

    step = MxUniverse.step
    """Alias of :meth:`mechanica.mechanica.MxUniverse.step`"""

    stop = MxUniverse.stop
    """Alias of :meth:`mechanica.mechanica.MxUniverse.stop`"""

    start = MxUniverse.start
    """Alias of :meth:`mechanica.mechanica.MxUniverse.start`"""

    # From MxUtils

    test = _MxTest

    random_points = MxRandomPoints
    """Alias of :meth:`mechanica.mechanica.MxRandomPoints`"""

    points = MxPoints
    """Alias of :meth:`mechanica.mechanica.MxPoints`"""

    primes = MxMath_FindPrimes
    cpuinfo = MxSystem.cpu_info()  #: :meta hide-value:
    compile_flags = MxSystem.compile_flags()  #: :meta hide-value:
%}
