/*
 * MxSystem.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXSYSTEM_H_
#define SRC_MXSYSTEM_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include <MxUtil.h>
#include <MxLogger.h>
#include "MxView.h"
#include <rendering/MxGlInfo.h>
#include <rendering/MxEglInfo.h>

void MxPrintPerformanceCounters();

HRESULT MxLoggerCallbackImpl(MxLogEvent, std::ostream *);

/**
 * @brief Provides various methods to interact with the 
 * rendering engine and host CPU. 
 * 
 */
struct CAPI_EXPORT MxSystem {

   const InstructionSet cpuInfo;
   const MxCompileFlags compileFlags;
   const MxGLInfo glInfo;
   const MxEGLInfo eglInfo;

   static std::tuple<char*, size_t> testImage();
   static std::tuple<char*, size_t> imageData();
   static bool contextHasCurrent();
   static HRESULT contextMakeCurrent();
   static HRESULT contextRelease();

   /**
    * @brief Set the camera view parameters
    * 
    * @param eye camera eye
    * @param center view center
    * @param up view upward direction
    * @return HRESULT 
    */
   static HRESULT cameraMoveTo(const MxVector3f &eye, const MxVector3f &center, const MxVector3f &up);

   /**
    * @brief Set the camera view parameters
    * 
    * @param center target camera view center position
    * @param rotation target camera rotation
    * @param zoom target camera zoom
    * @return HRESULT 
    */
   static HRESULT cameraMoveTo(const MxVector3f &center, const MxQuaternionf &rotation, const float &zoom);

   /**
    * @brief Move the camera to view the domain from the bottm
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewBottom();

   /**
    * @brief Move the camera to view the domain from the top
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewTop();
   
   /**
    * @brief Move the camera to view the domain from the left
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewLeft();

   /**
    * @brief Move the camera to view the domain from the right
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewRight();

   /**
    * @brief Move the camera to view the domain from the back
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewBack();

   /**
    * @brief Move the camera to view the domain from the front
    * 
    * @return HRESULT 
    */
   static HRESULT cameraViewFront();

   /**
    * @brief Reset the camera
    * 
    * @return HRESULT 
    */
   static HRESULT cameraReset();

   /* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
   static HRESULT cameraRotateMouse(const MxVector2i &mousePos);

   /* Translate the camera from the previous (screen) mouse position to
   the current (screen) mouse position */
   static HRESULT cameraTranslateMouse(const MxVector2i &mousePos);

   /* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
   static HRESULT cameraInitMouse(const MxVector2i &mousePos);
   
   /* Translate the camera by the delta amount of (NDC) mouse position.
   Note that NDC position must be in [-1, -1] to [1, 1]. */
   static HRESULT cameraTranslateBy(const MxVector2f &trans);
   
   /**
    * @brief Zoom the camera by an increment in distance. 
    * 
    * Positive values zoom in. 
    * 
    * @param delta zoom increment
    * @return HRESULT 
    */
   static HRESULT cameraZoomBy(const float &delta);
   
   /**
    * @brief Zoom the camera to a distance. 
    * 
    * @param distance zoom distance
    * @return HRESULT 
    */
   static HRESULT cameraZoomTo(const float &distance);
   
   /**
    * @brief Rotate the camera to a point from the view center a distance along an axis. 
    * 
    * Only rotates the view to the given eye position.
    * 
    * @param axis axis from the view center
    * @param distance distance along the axis
    * @return HRESULT 
    */
   static HRESULT cameraRotateToAxis(const MxVector3f &axis, const float &distance);

   /**
    * @brief Rotate the camera to a set of Euler angles. 
    * 
    * Rotations are Z-Y-X. 
    * 
    * @param angles 
    * @return HRESULT 
    */
   static HRESULT cameraRotateToEulerAngle(const MxVector3f &angles);

   /**
    * @brief Rotate the camera by a set of Euler angles. 
    * 
    * Rotations are Z-Y-X. 
    * 
    * @param angles 
    * @return HRESULT 
    */
   static HRESULT cameraRotateByEulerAngle(const MxVector3f &angles);

   /**
    * @brief Get the current camera view center position
    * 
    * @return MxVector3f 
    */
   static MxVector3f cameraCenter();

   /**
    * @brief Get the current camera rotation
    * 
    * @return MxQuaternionf 
    */
   static MxQuaternionf cameraRotation();

   /**
    * @brief Get the current camera zoom
    * 
    * @return float 
    */
   static float cameraZoom();
    
   /**
    * @brief Get the universe renderer
    * 
    * @return struct MxUniverseRenderer* 
    */
   static struct MxUniverseRenderer *getRenderer();
   
   /* Update screen size after the window has been resized */
   static HRESULT viewReshape(const MxVector2i &windowSize);
   static std::string performanceCounters();

   /**
    * @brief Get CPU info
    * 
    * @return std::unordered_map<std::string, bool> 
    */
   static std::unordered_map<std::string, bool> cpu_info();

   /**
    * @brief Get compiler flags of this installation
    * 
    * @return std::list<std::string> 
    */
   static std::list<std::string> compile_flags();

   /**
    * @brief Get OpenGL info
    * 
    * @return std::unordered_map<std::string, std::string> 
    */
   static std::unordered_map<std::string, std::string> gl_info();

   /**
    * @brief Get EGL info
    * 
    * @return std::string 
    */
   static std::string egl_info();
   static std::unordered_map<std::string, std::string> test_headless();

public:

   MxSystem() {};
   ~MxSystem() {};

};

struct CAPI_EXPORT MxSystemPy : MxSystem {

public:
   MxSystemPy() {};
   ~MxSystemPy() {};

   static PyObject *test_image();
   static PyObject *image_data();

   /**
    * @brief Test whether Mechanica is running in an interactive terminal
    * 
    * @return true if running in an interactive terminal
    * @return false 
    */
   static bool is_terminal_interactive();

   /**
    * @brief Test whether Mechanica is running in a Jupyter notebook
    * 
    * @return true if running in a Jupyter notebook
    * @return false 
    */
   static bool is_jupyter_notebook();

   static PyObject *jwidget_init(PyObject *args, PyObject *kwargs);
   static PyObject *jwidget_run(PyObject *args, PyObject *kwargs);
};

#endif /* SRC_MXSYSTEM_H_ */
