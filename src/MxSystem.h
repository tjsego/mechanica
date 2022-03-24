/*
 * MxSystem.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXSYSTEM_H_
#define SRC_MXSYSTEM_H_

#include "mechanica_private.h"
#include <MxUtil.h>
#include <MxLogger.h>
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

   /**
    * @brief Save a screenshot of the current scene
    * 
    * @param filePath path of file to save
    * @return HRESULT 
    */
   static HRESULT screenshot(const std::string &filePath);
   
   /**
    * @brief Save a screenshot of the current scene
    * 
    * @param filePath path of file to save
    * @param decorate flag to decorate the scene in the screenshot
    * @param bgcolor background color of the scene in the screenshot
    * @return HRESULT 
    */
   static HRESULT screenshot(const std::string &filePath, const bool &decorate, const MxVector3f &bgcolor);
   
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

   /**
    * @brief Translate the camera down
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateDown();

   /**
    * @brief Translate the camera up
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateUp();

   /**
    * @brief Translate the camera right
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateRight();

   /**
    * @brief Translate the camera left
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateLeft();
   
   /**
    * @brief Translate the camera forward
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateForward();
   
   /**
    * @brief Translate the camera backward
    * 
    * @return HRESULT 
    */
   static HRESULT cameraTranslateBackward();
   
   /**
    * @brief Rotate the camera down
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRotateDown();
   
   /**
    * @brief Rotate the camera up
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRotateUp();
   
   /**
    * @brief Rotate the camera left
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRotateLeft();
   
   /**
    * @brief Rotate the camera right
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRotateRight();
   
   /**
    * @brief Roll the camera left
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRollLeft();
   
   /**
    * @brief Rotate the camera right
    * 
    * @return HRESULT 
    */
   static HRESULT cameraRollRight();

   /**
    * @brief Zoom the camera in
    * 
    * @return HRESULT 
    */
   static HRESULT cameraZoomIn();

   /**
    * @brief Zoom the camera out
    * 
    * @return HRESULT 
    */
   static HRESULT cameraZoomOut();

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

   /**
    * @brief Get the ambient color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getAmbientColor();

   /**
    * @brief Set the ambient color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setAmbientColor(const MxVector3f &color);

   /**
    * @brief Set the ambient color of a subrenderer
    * 
    * @param color 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setAmbientColor(const MxVector3f &color, const unsigned int &srFlag);

   /**
    * @brief Get the diffuse color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getDiffuseColor();

   /**
    * @brief Set the diffuse color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setDiffuseColor(const MxVector3f &color);

   /**
    * @brief Set the diffuse color of a subrenderer
    * 
    * @param color 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setDiffuseColor(const MxVector3f &color, const unsigned int &srFlag);

   /**
    * @brief Get specular color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getSpecularColor();

   /**
    * @brief Set the specular color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setSpecularColor(const MxVector3f &color);

   /**
    * @brief Set the specular color of a subrenderer
    * 
    * @param color 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setSpecularColor(const MxVector3f &color, const unsigned int &srFlag);

   /**
    * @brief Get the shininess
    * 
    * @return float 
    */
   static float getShininess();

   /**
    * @brief Set the shininess
    * 
    * @param shininess 
    * @return HRESULT 
    */
   static HRESULT setShininess(const float &shininess);

   /**
    * @brief Set the shininess of a subrenderer
    * 
    * @param shininess 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setShininess(const float &shininess, const unsigned int &srFlag);

   /**
    * @brief Get the grid color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getGridColor();

   /**
    * @brief Set the grid color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setGridColor(const MxVector3f &color);

   /**
    * @brief Get the scene box color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getSceneBoxColor();

   /**
    * @brief Set the scene box color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setSceneBoxColor(const MxVector3f &color);

   /**
    * @brief Get the light direction
    * 
    * @return MxVector3f 
    */
   static MxVector3f getLightDirection();

   /**
    * @brief Set the light direction
    * 
    * @param lightDir 
    * @return HRESULT 
    */
   static HRESULT setLightDirection(const MxVector3f& lightDir);

   /**
    * @brief Set the light direction of a subrenderer
    * 
    * @param lightDir 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setLightDirection(const MxVector3f& lightDir, const unsigned int &srFlag);

   /**
    * @brief Get the light color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getLightColor();

   /**
    * @brief Set the light color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setLightColor(const MxVector3f &color);

   /**
    * @brief Set the light color of a subrenderer
    * 
    * @param color 
    * @param srFlag 
    * @return HRESULT 
    */
   static HRESULT setLightColor(const MxVector3f &color, const unsigned int &srFlag);

   /**
    * @brief Get the background color
    * 
    * @return MxVector3f 
    */
   static MxVector3f getBackgroundColor();

   /**
    * @brief Set the background color
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setBackgroundColor(const MxVector3f &color);

   /**
    * @brief Test whether the rendered scene is decorated
    * 
    * @return true 
    * @return false 
    */
   static bool decorated();

   /**
    * @brief Set flag to draw/not draw scene decorators (e.g., grid)
    * 
    * @param decorate flag; true says to decorate
    * @return HRESULT 
    */
   static HRESULT decorateScene(const bool &decorate);

   /**
    * @brief Test whether discretization is current shown
    * 
    * @return true 
    * @return false 
    */
   static bool showingDiscretization();

   /**
    * @brief Set flag to draw/not draw discretization
    * 
    * @param show flag; true says to show
    * @return HRESULT 
    */
   static HRESULT showDiscretization(const bool &show);

   /**
    * @brief Get the current color of the discretization grid
    * 
    * @return MxVector3f 
    */
   static MxVector3f getDiscretizationColor();

   /**
    * @brief Set the color of the discretization grid
    * 
    * @param color 
    * @return HRESULT 
    */
   static HRESULT setDiscretizationColor(const MxVector3f &color);
   
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

#endif /* SRC_MXSYSTEM_H_ */
