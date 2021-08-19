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

   /* Set the camera view parameters: eye position, view center, up
   direction */
   static HRESULT cameraMoveTo(const MxVector3f *eye=NULL, const MxVector3f *center=NULL, const MxVector3f *up=NULL);
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
   
   /* Zoom the camera (positive delta = zoom in, negative = zoom out) */
   static HRESULT cameraZoomBy(const float &delta);
   
   /* Zoom the camera (positive delta = zoom in, negative = zoom out) */
   static HRESULT cameraZoomTo(const float &distance);
   
   /*
   * Set the camera view parameters: eye position, view center, up
   * direction, only rotates the view to the given eye position.
   */
   static HRESULT cameraRotateToAxis(const MxVector3f &axis, const float &distance);
   static HRESULT cameraRotateToEulerAngle(const MxVector3f &angles);
   static HRESULT cameraRotateByEulerAngle(const MxVector3f &angles);
   
   /* Update screen size after the window has been resized */
   static HRESULT viewReshape(const MxVector2i &windowSize);
   static std::string performanceCounters();

   static std::unordered_map<std::string, bool> cpu_info();
   static std::list<std::string> compile_flags();
   static std::unordered_map<std::string, std::string> gl_info();
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
   static bool is_terminal_interactive();
   static bool is_jupyter_notebook();
   static PyObject *jwidget_init(PyObject *args, PyObject *kwargs);
   static PyObject *jwidget_run(PyObject *args, PyObject *kwargs);
};

CAPI_FUNC(MxSystem*) getSystem();
CAPI_FUNC(MxSystemPy*) getSystemPy();

#endif /* SRC_MXSYSTEM_H_ */
