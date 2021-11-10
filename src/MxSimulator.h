/*
 * MxSimulator.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef SRC_MXSIMULATOR_H_
#define SRC_MXSIMULATOR_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"
#ifdef MX_WITHCUDA
#include "cuda/MxSimulatorCUDAConfig.h"
#endif

#include "Magnum/Platform/GLContext.h"
#include "Magnum/Platform/Implementation/DpiScaling.h"
#include "MxUniverse.h"

class MxGlfwWindow;

// Of the available integrator types, these are supported by MxSimulator
enum class MxSimulator_EngineIntegrator : int {
    FORWARD_EULER = EngineIntegrator::FORWARD_EULER,
    RUNGE_KUTTA_4 = EngineIntegrator::RUNGE_KUTTA_4
};

enum MxSimulator_Key {
    MXSIMULATOR_NONE,
    MXSIMULATOR_WINDOWLESS,
    MXSIMULATOR_GLFW
};

// struct MxSimulator_ConfigurationItem {
//     uint32_t key;
//     union {
//         int intVal;
//         int intVecVal[4];
//     };
// };

enum MxSimulator_Options {
    Windowless = 1 << 0,

    GLFW = 1 << 1,
    /**
     * Forward compatible context
     *
     * @requires_gl Core/compatibility profile distinction and forward
     *      compatibility applies only to desktop GL.
     */
    GlForwardCompatible = 1 << 2,

    /**
     * Specifies whether errors should be generated by the context.
     * If enabled, situations that would have generated errors instead
     * cause undefined behavior.
     *
     * @note Supported since GLFW 3.2.
     */
    GlNoError = 1 << 3,


    /**
     * Debug context. Enabled automatically if the
     * `--magnum-gpu-validation` @ref GL-Context-command-line "command-line option"
     * is present.
     */
    GlDebug = 1 << 4,

    GlStereo = 1 << 5     /**< Stereo rendering */
};

struct MxSimulator;

enum class MxSimulator_DpiScalingPolicy : UnsignedByte {
    /* Using 0 for an "unset" value */

    #ifdef CORRADE_TARGET_APPLE
    Framebuffer = 1,
    #endif

    #ifndef CORRADE_TARGET_APPLE
    Virtual = 2,

    Physical = 3,
    #endif

    Default
        #ifdef CORRADE_TARGET_APPLE
        = Framebuffer
        #else
        = Virtual
        #endif
};

struct CAPI_EXPORT MxSimulator_Config
{
public:

    /**
     * @brief DPI scaling policy
     *
     * DPI scaling policy when requesting a particular window size. Can
     * be overriden on command-line using `--magnum-dpi-scaling` or via
     * the `MAGNUM_DPI_SCALING` environment variable.
     * @see @ref setSize(), @ref Platform-Sdl2Application-dpi
     */


    /*implicit*/
    MxSimulator_Config();

    ~MxSimulator_Config() {};

    /** @brief Window title */
    std::string title() const
    {
        return _title;
    }

    /**
     * @brief Set window title
     * @return Reference to self (for method chaining)
     *
     * Default is @cpp "Magnum GLFW Application" @ce.
     */
    void setTitle(std::string title)
    {
        _title = std::move(title);
    }

    /** @brief Window size */
    MxVector2i windowSize() const
    {
        return _size;
    }

    /**
     * @brief DPI scaling policy
     *
     * If @ref dpiScaling() is non-zero, it has a priority over this value.
     * The `--magnum-dpi-scaling` command-line option has a priority over
     * any application-set value.
     * @see @ref setSize(const MxVector2i&, DpiScalingPolicy)
     */
    MxSimulator_DpiScalingPolicy dpiScalingPolicy() const
    {
        return _dpiScalingPolicy;
    }

    /**
     * @brief Custom DPI scaling
     *
     * If zero, then @ref dpiScalingPolicy() has a priority over this
     * value. The `--magnum-dpi-scaling` command-line option has a priority
     * over any application-set value.
     * @see @ref setSize(const MxVector2i&, const Vector2&)
     * @todo change this on a DPI change event (GLFW 3.3 has a callback:
     *  https://github.com/mosra/magnum/issues/243#issuecomment-388384089)
     */
    MxVector2f dpiScaling() const
    {
        return _dpiScaling;
    }

    void setDpiScaling(const MxVector2f &vec)
        {
            _dpiScaling = vec;
        }


    void setSizeAndScaling(const MxVector2i& size, MxSimulator_DpiScalingPolicy dpiScalingPolicy = MxSimulator_DpiScalingPolicy::Default) {
                _size = size;
                _dpiScalingPolicy = dpiScalingPolicy;

            }


    void setSizeAndScaling(const MxVector2i& size, const MxVector2f& dpiScaling) {
                _size = size;
                _dpiScaling = dpiScaling;
    }

    /**
     * @brief Set window size
     * @param size              Desired window size
     * @param dpiScalingPolicy  Policy based on which DPI scaling will be set
     * @return Reference to self (for method chaining)
     *
     * Default is @cpp {800, 600} @ce. See @ref Platform-MxGlfwApplication-dpi
     * for more information.
     * @see @ref setSize(const MxVector2i&, const MxVector2&)
     */
    void setWindowSize(const MxVector2i &size)
    {
        _size = size;
    }

    /** @brief Window flags */
    uint32_t windowFlags() const
    {
        return _windowFlags;
    }

    /**
     * @brief Set window flags
     * @return  Reference to self (for method chaining)
     *
     * Default is @ref WindowFlag::Focused.
     */
    void setWindowFlags(uint32_t windowFlags)
    {
        _windowFlags = windowFlags;
    }

    bool windowless() const {
        return _windowless;
    }

    void setWindowless(bool val) {
        _windowless = val;
    }

    int size() const {
        return universeConfig.nParticles;
    }

    void setSize(int i ) {
        universeConfig.nParticles = i;
    }

    MxUniverseConfig universeConfig;

    int queues;

    int argc = 0;

    char** argv = NULL;
    
    
    std::vector<MxVector4f> clipPlanes;

private:
    std::string _title;
    MxVector2i _size;
    uint32_t _windowFlags;
    MxSimulator_DpiScalingPolicy _dpiScalingPolicy;
    MxVector2f _dpiScaling;
    bool _windowless;
};

/**
 * @brief The Simulator is the entry point to simulation, this is the very first object
 * that needs to be initialized  before any other method can be called. All the
 * methods of the Simulator are static, but the constructor needs to be called
 * first to initialize everything.
 *
 * The Simulator manages all of the operating system interface, it manages
 * window creation, end user input events, GPU access, threading, inter-process
 * messaging and so forth. All 'physical' modeling concepts go in the Universe.
 */
struct CAPI_EXPORT MxSimulator {

    class CAPI_EXPORT GLConfig;

    /**
     * @brief Window flag
     *
     * @see @ref WindowFlags, @ref setWindowFlags()
     */
    enum WindowFlags : UnsignedShort
    {
        /** Fullscreen window */
        Fullscreen = 1 << 0,

        /**
         * No window decoration
         */
        Borderless = 1 << 1,

        Resizable = 1 << 2,    /**< Resizable window */
        Hidden = 1 << 3,       /**< Hidden window */


        /**
         * Maximized window
         *
         * @note Supported since GLFW 3.2.
         */
        Maximized = 1 << 4,


        Minimized = 1 << 5,    /**< Minimized window */

        /**
         * Always on top
         * @m_since_latest
         */
        AlwaysOnTop = 1 << 6,



        /**
         * Automatically iconify (minimize) if fullscreen window loses
         * input focus
         */
        AutoIconify = 1 << 7,

        /**
         * Window has input focus
         *
         * @todo there's also GLFW_FOCUS_ON_SHOW, what's the difference?
         */
        Focused = 1 << 8,

        /**
         * Do not create any GPU context. Use together with
         * @ref GlfwApplication(const Arguments&),
         * @ref GlfwApplication(const Arguments&, const Configuration&),
         * @ref create(const Configuration&) or
         * @ref tryCreate(const Configuration&) to prevent implicit
         * creation of an OpenGL context.
         *
         * @note Supported since GLFW 3.2.
         */
        Contextless = 1 << 9

    };
    
    struct MxUniverseRenderer *getRenderer();


    int32_t kind;
    struct MxApplication *app;

    enum Flags {
        Running = 1 << 0
    };

    /**
     * gets the global simulator object, throws exception if fail.
     */
    static MxSimulator *get();

    static HRESULT initConfig(const MxSimulator_Config &conf, const GLConfig &glConf);

    /**
     * This function processes only those events that are already in the event
     * queue and then returns immediately. Processing events will cause the window
     * and input callbacks associated with those events to be called.
     *
     * On some platforms, a window move, resize or menu operation will cause
     * event processing to block. This is due to how event processing is designed
     * on those platforms. You can use the window refresh callback to redraw the
     * contents of your window when necessary during such operations.
     */
    static HRESULT pollEvents();

    /**
     *   This function puts the calling thread to sleep until at least one
     *   event is available in the event queue. Once one or more events are
     *   available, it behaves exactly like glfwPollEvents, i.e. the events
     *   in the queue are processed and the function then returns immediately.
     *   Processing events will cause the window and input callbacks associated
     *   with those events to be called.
     *
     *   Since not all events are associated with callbacks, this function may return
     *   without a callback having been called even if you are monitoring all callbacks.
     *
     *  On some platforms, a window move, resize or menu operation will cause event
     *  processing to block. This is due to how event processing is designed on
     *  those platforms. You can use the window refresh callback to redraw the
     *  contents of your window when necessary during such operations.
     */
    static HRESULT waitEvents();

    /**
     * This function puts the calling thread to sleep until at least
     * one event is available in the event queue, or until the specified
     * timeout is reached. If one or more events are available, it behaves
     * exactly like pollEvents, i.e. the events in the queue are
     * processed and the function then returns immediately. Processing
     * events will cause the window and input callbacks associated with those
     * events to be called.
     *
     * The timeout value must be a positive finite number.
     * Since not all events are associated with callbacks, this function may
     * return without a callback having been called even if you are monitoring
     * all callbacks.
     *
     * On some platforms, a window move, resize or menu operation will cause
     * event processing to block. This is due to how event processing is designed
     * on those platforms. You can use the window refresh callback to redraw the
     * contents of your window when necessary during such operations.
     */
    static HRESULT waitEventsTimeout(double  timeout);

    /**
     * This function posts an empty event from the current thread
     * to the event queue, causing waitEvents or waitEventsTimeout to return.
     */
    static HRESULT postEmptyEvent();

    /**
     * @brief Runs the event loop until all windows close or simulation time expires. 
     * Automatically performs universe time propogation. 
     * 
     * @param et final time; a negative number runs infinitely
     * @return HRESULT 
     */
    static HRESULT run(double et);

    /**
     * @brief Shows any windows that were specified in the config. 
     * Works just like MatPlotLib's ``show`` method. 
     * The ``show`` method does not start the universe time propagation unlike ``run`` and ``irun``.
     * 
     * @return HRESULT 
     */
    static HRESULT show();

    /**
     * @brief Closes the main window, while the application / simulation continues to run.
     * 
     * @return HRESULT 
     */
    static HRESULT close();

    static HRESULT destroy();

    static HRESULT redraw();

    /**
     * This function sets the swap interval for the current OpenGL or OpenGL ES context, i.e. the number of screen updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning. This is sometimes called vertical synchronization, vertical retrace synchronization or just vsync.
     * 
     * A context that supports either of the WGL_EXT_swap_control_tear and GLX_EXT_swap_control_tear extensions also accepts negative swap intervals, which allows the driver to swap immediately even if a frame arrives a little bit late. You can check for these extensions with glfwExtensionSupported.
     * 
     * A context must be current on the calling thread. Calling this function without a current context will cause a GLFW_NO_CURRENT_CONTEXT error.
     * 
     * This function does not apply to Vulkan. If you are rendering with Vulkan, see the present mode of your swapchain instead.
     * 
     * Parameters
     * [in]    interval    The minimum number of screen updates to wait for until the buffers are swapped by glfwSwapBuffers.
     * Errors
     * Possible errors include GLFW_NOT_INITIALIZED, GLFW_NO_CURRENT_CONTEXT and GLFW_PLATFORM_ERROR.
     * Remarks
     * This function is not called during context creation, leaving the swap interval set to whatever is the default on that platform. This is done because some swap interval extensions used by GLFW do not allow the swap interval to be reset to zero once it has been set to a non-zero value.
     * Some GPU drivers do not honor the requested swap interval, either because of a user setting that overrides the application's request or due to bugs in the driver.
     */
    static HRESULT swapInterval(int si);

    static const int getNumThreads();

    static const MxGlfwWindow *getWindow();

    #ifdef MX_WITHCUDA
    static MxSimulatorCUDAConfig *getCUDAConfig();
    #endif
    
    // list of windows.
    std::vector<MxGlfwWindow*> windows;
};


/**
 * main simulator init method
 */
CAPI_FUNC(HRESULT) MxSimulator_init(const std::vector<std::string> &argv);

/**
 * main simulator init method
 */
CAPI_FUNC(HRESULT) MxSimulator_initC(const MxSimulator_Config &conf, const std::vector<std::string> &appArgv=std::vector<std::string>());


/**
 @brief OpenGL context configuration

 The created window is always with a double-buffered OpenGL context.

 @note This function is available only if Magnum is compiled with
 @ref MAGNUM_TARGET_GL enabled (done by default). See @ref building-features
 for more information.

 @see @ref MxGlfwApplication(), @ref create(), @ref tryCreate()
 */
class MxSimulator::GLConfig {
public:
    /**
     * @brief Context flag
     *
     * @see @ref Flags, @ref setFlags(), @ref GL::Context::Flag
     */
    enum  Flag: uint32_t {
#ifndef MAGNUM_TARGET_GLES
        /**
         * Forward compatible context
         *
         * @requires_gl Core/compatibility profile distinction and forward
         *      compatibility applies only to desktop GL.
         */
        ForwardCompatible = 1 << 0,
#endif

#if defined(DOXYGEN_GENERATING_OUTPUT) || defined(GLFW_CONTEXT_NO_ERROR)
        /**
         * Specifies whether errors should be generated by the context.
         * If enabled, situations that would have generated errors instead
         * cause undefined behavior.
         *
         * @note Supported since GLFW 3.2.
         */
        NoError = 1 << 1,
#endif

        /**
         * Debug context. Enabled automatically if the
         * `--magnum-gpu-validation` @ref GL-Context-command-line "command-line option"
         * is present.
         */
        Debug = 1 << 2,

        Stereo = 1 << 3     /**< Stereo rendering */
    };

    /**
     * @brief Context flags
     *
     * @see @ref setFlags(), @ref GL::Context::Flags
     */
    typedef uint32_t Flags;

    explicit GLConfig();
    ~GLConfig();

    /** @brief Context flags */
    Flags flags() const { return _flags; }

    /**
     * @brief Set context flags
     * @return Reference to self (for method chaining)
     *
     * Default is @ref Flag::ForwardCompatible on desktop GL and no flags
     * on OpenGL ES.
     * @see @ref addFlags(), @ref clearFlags(), @ref GL::Context::flags()
     */
    GLConfig& setFlags(Flags flags) {
        _flags = flags;
        return *this;
    }

    /**
     * @brief Add context flags
     * @return Reference to self (for method chaining)
     *
     * Unlike @ref setFlags(), ORs the flags with existing instead of
     * replacing them. Useful for preserving the defaults.
     * @see @ref clearFlags()
     */
    GLConfig& addFlags(Flags flags) {
        _flags |= flags;
        return *this;
    }

    /**
     * @brief Clear context flags
     * @return Reference to self (for method chaining)
     *
     * Unlike @ref setFlags(), ANDs the inverse of @p flags with existing
     * instead of replacing them. Useful for removing default flags.
     * @see @ref addFlags()
     */
    GLConfig& clearFlags(Flags flags) {
        _flags &= ~flags;
        return *this;
    }

    /** @brief Context version */
    GL::Version version() const { return _version; }

    /**
     * @brief Set context version
     *
     * If requesting version greater or equal to OpenGL 3.2, core profile
     * is used. The created context will then have any version which is
     * backwards-compatible with requested one. Default is
     * @ref GL::Version::None, i.e. any provided version is used.
     */
    GLConfig& setVersion(GL::Version version) {
        _version = version;
        return *this;
    }

    /** @brief Color buffer size */
    MxVector4i colorBufferSize() const { return _colorBufferSize; }

    /**
     * @brief Set color buffer size
     *
     * Default is @cpp {8, 8, 8, 0} @ce (8-bit-per-channel RGB, no alpha).
     * @see @ref setDepthBufferSize(), @ref setStencilBufferSize()
     */
    GLConfig& setColorBufferSize(const MxVector4i& size) {
        _colorBufferSize = size;
        return *this;
    }

    /** @brief Depth buffer size */
    Int depthBufferSize() const { return _depthBufferSize; }

    /**
     * @brief Set depth buffer size
     *
     * Default is @cpp 24 @ce bits.
     * @see @ref setColorBufferSize(), @ref setStencilBufferSize()
     */
    GLConfig& setDepthBufferSize(Int size) {
        _depthBufferSize = size;
        return *this;
    }

    /** @brief Stencil buffer size */
    Int stencilBufferSize() const { return _stencilBufferSize; }

    /**
     * @brief Set stencil buffer size
     *
     * Default is @cpp 0 @ce bits (i.e., no stencil buffer).
     * @see @ref setColorBufferSize(), @ref setDepthBufferSize()
     */
    GLConfig& setStencilBufferSize(Int size) {
        _stencilBufferSize = size;
        return *this;
    }

    /** @brief Sample count */
    Int sampleCount() const { return _sampleCount; }

    /**
     * @brief Set sample count
     * @return Reference to self (for method chaining)
     *
     * Default is @cpp 0 @ce, thus no multisampling. The actual sample
     * count is ignored, GLFW either enables it or disables. See also
     * @ref GL::Renderer::Feature::Multisampling.
     */
    GLConfig& setSampleCount(Int count) {
        _sampleCount = count;
        return *this;
    }

    /** @brief sRGB-capable default framebuffer */
    bool isSrgbCapable() const { return _srgbCapable; }

    /**
     * @brief Set sRGB-capable default framebuffer
     *
     * Default is @cpp false @ce. See also
     * @ref GL::Renderer::Feature::FramebufferSrgb.
     * @return Reference to self (for method chaining)
     */
    GLConfig& setSrgbCapable(bool enabled) {
        _srgbCapable = enabled;
        return *this;
    }


private:
    MxVector4i _colorBufferSize;
    Int _depthBufferSize, _stencilBufferSize;
    Int _sampleCount;
    GL::Version _version;
    Flags _flags;
    bool _srgbCapable;
};

struct CAPI_EXPORT MxSimulatorPy : MxSimulator {

public:

    /**
     * gets the global simulator object, throws exception if fail.
     */
    static MxSimulatorPy *get();

    static PyObject *_run(PyObject *args, PyObject *kwargs);
    
    /**
     * @brief Interactive python version of the run loop. This checks the ipython context and lets 
     * ipython process keyboard input, while we also running the simulator and processing window messages.
     * 
     * @return HRESULT 
     */
    static HRESULT irun();

    static HRESULT _show();

    static void *wait_events(const double &timeout=-1);

};

CAPI_FUNC(HRESULT) _setIPythonInputHook(PyObject *_ih);

CAPI_FUNC(HRESULT) _onIPythonNotReady();

/**
 * @brief Initialize a simulation in Python
 * 
 * @param args positional arguments; first argument is name of simulation (if any)
 * @param kwargs keyword arguments; currently supported are
 * 
 *      dim: (3-component list of floats) the dimensions of the spatial domain; default is [10., 10., 10.]
 * 
 *      cutoff: (float) simulation cutoff distance; default is 1.
 * 
 *      cells: (3-component list of ints) the discretization of the spatial domain; default is [4, 4, 4]
 * 
 *      threads: (int) number of threads; default is hardware maximum
 * 
 *      integrator: (int) simulation integrator; default is FORWARD_EULER
 * 
 *      dt: (float) time discretization; default is 0.01
 * 
 *      bc: (int or dict) boundary conditions; default is everywhere periodic
 * 
 *      window_size: (2-component list of ints) size of application window; default is [800, 600]
 * 
 *      logger_level: (int) logger level; default is no logging
 * 
 *      clip_planes: (list of tuple of (MxVector3f, MxVector3f)) list of point-normal pairs of clip planes; default is no planes
 */
CAPI_FUNC(PyObject *) MxSimulatorPy_init(PyObject *args, PyObject *kwargs);

// const MxVector3 &origin, const MxVector3 &dim,
// int nParticles, double dt = 0.005, float temp = 100

CAPI_FUNC(int) universe_init(const MxUniverseConfig &conf);

CAPI_FUNC(HRESULT) modules_init();

#endif /* SRC_MXSIMULATOR_H_ */
