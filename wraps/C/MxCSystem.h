/**
 * @file MxCSystem.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines C support for MxSystem
 * @date 2022-04-04
 */

#ifndef _WRAPS_C_MXCSYSTEM_H_
#define _WRAPS_C_MXCSYSTEM_H_

#include <mx_port.h>


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the current image data
 * 
 * @param imgData image data
 * @param imgSize image size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_imageData(char **imgData, size_t *imgSize);

/**
* @brief Save a screenshot of the current scene
* 
* @param filePath path of file to save
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_screenshot(const char *filePath);

/**
* @brief Save a screenshot of the current scene
* 
* @param filePath path of file to save
* @param decorate flag to decorate the scene in the screenshot
* @param bgcolor background color of the scene in the screenshot
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_screenshotS(const char *filePath, bool decorate, float *bgcolor);

/**
 * @brief Test whether the context is current
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_contextHasCurrent(bool *current);

/**
 * @brief Make the context current
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_contextMakeCurrent();

/**
 * @brief Release the current context
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_contextRelease();

/**
* @brief Set the camera view parameters
* 
* @param eye camera eye
* @param center view center
* @param up view upward direction
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraMoveTo(float *eye, float *center, float *up);

/**
* @brief Set the camera view parameters
* 
* @param center target camera view center position
* @param rotation target camera rotation
* @param zoom target camera zoom
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraMoveToR(float *center, float *rotation, float zoom);

/**
* @brief Move the camera to view the domain from the bottm
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewBottom();

/**
* @brief Move the camera to view the domain from the top
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewTop();

/**
* @brief Move the camera to view the domain from the left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewLeft();

/**
* @brief Move the camera to view the domain from the right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewRight();

/**
* @brief Move the camera to view the domain from the back
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewBack();

/**
* @brief Move the camera to view the domain from the front
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraViewFront();

/**
* @brief Reset the camera
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraReset();

/**
 * @brief Rotate the camera from the previous (screen) mouse position to the current (screen) position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordinate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateMouse(int x, int y);

/**
 * @brief Translate the camera from the previous (screen) mouse position to the current (screen) mouse position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordintae
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateMouse(int x, int y);

/**
* @brief Translate the camera down
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateDown();

/**
* @brief Translate the camera up
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateUp();

/**
* @brief Translate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateRight();

/**
* @brief Translate the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateLeft();

/**
* @brief Translate the camera forward
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateForward();

/**
* @brief Translate the camera backward
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateBackward();

/**
* @brief Rotate the camera down
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateDown();

/**
* @brief Rotate the camera up
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateUp();

/**
* @brief Rotate the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateLeft();

/**
* @brief Rotate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateRight();

/**
* @brief Roll the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRollLeft();

/**
* @brief Rotate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRollRight();

/**
* @brief Zoom the camera in
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraZoomIn();

/**
* @brief Zoom the camera out
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraZoomOut();

/**
 * @brief Initialize the camera at a mouse position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordinate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraInitMouse(int x, int y);

/**
 * @brief Translate the camera by the delta amount of (NDC) mouse position. 
 * 
 * Note that NDC position must be in [-1, -1] to [1, 1].
 * 
 * @param x horizontal delta
 * @param y vertical delta
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraTranslateBy(float x, float y);

/**
* @brief Zoom the camera by an increment in distance. 
* 
* Positive values zoom in. 
* 
* @param delta zoom increment
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraZoomBy(float delta);

/**
* @brief Zoom the camera to a distance. 
* 
* @param distance zoom distance
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraZoomTo(float distance);

/**
* @brief Rotate the camera to a point from the view center a distance along an axis. 
* 
* Only rotates the view to the given eye position.
* 
* @param axis axis from the view center
* @param distance distance along the axis
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateToAxis(float *axis, float distance);

/**
* @brief Rotate the camera to a set of Euler angles. 
* 
* Rotations are Z-Y-X. 
* 
* @param angles 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateToEulerAngle(float *angles);

/**
* @brief Rotate the camera by a set of Euler angles. 
* 
* Rotations are Z-Y-X. 
* 
* @param angles 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_cameraRotateByEulerAngle(float *angles);

/**
 * @brief Get the current camera view center position
 * 
 * @param center camera center
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraCenter(float **center);

/**
 * @brief Get the current camera rotation
 * 
 * @param rotation camera rotation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraRotation(float **rotation);

/**
 * @brief Get the current camera zoom
 * 
 * @param zoom camera zoom
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_cameraZoom(float *zoom);

/**
 * @brief Get the ambient color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getAmbientColor(float **color);

/**
 * @brief Set the ambient color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_setAmbientColor(float *color);

/**
 * @brief Get the diffuse color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getDiffuseColor(float **color);

/**
* @brief Set the diffuse color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setDiffuseColor(float *color);

/**
 * @brief Get specular color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getSpecularColor(float **color);

/**
* @brief Set the specular color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setSpecularColor(float *color);

/**
 * @brief Get the shininess
 * 
 * @param shininess 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getShininess(float *shininess);

/**
* @brief Set the shininess
* 
* @param shininess 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setShininess(float shininess);

/**
 * @brief Get the grid color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getGridColor(float **color);

/**
* @brief Set the grid color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setGridColor(float *color);

/**
 * @brief Get the scene box color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getSceneBoxColor(float **color);

/**
* @brief Set the scene box color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setSceneBoxColor(float *color);

/**
 * @brief Get the light direction
 * 
 * @param lightDir 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getLightDirection(float **lightDir);

/**
* @brief Set the light direction
* 
* @param lightDir 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setLightDirection(float *lightDir);

/**
 * @brief Get the light color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getLightColor(float **color);

/**
* @brief Set the light color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setLightColor(float *color);

/**
 * @brief Get the background color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getBackgroundColor(float **color);

/**
* @brief Set the background color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setBackgroundColor(float *color);

/**
 * @brief Test whether the rendered scene is decorated
 * 
 * @param decorated 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_decorated(bool *decorated);

/**
* @brief Set flag to draw/not draw scene decorators (e.g., grid)
* 
* @param decorate flag; true says to decorate
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_decorateScene(bool decorate);

/**
 * @brief Test whether discretization is currently shown
 * 
 * @param showing 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_showingDiscretization(bool *showing);

/**
* @brief Set flag to draw/not draw discretization
* 
* @param show flag; true says to show
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_showDiscretization(bool show);

/**
 * @brief Get the current color of the discretization grid
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getDiscretizationColor(float **color);

/**
* @brief Set the color of the discretization grid
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) MxCSystem_setDiscretizationColor(float *color);

/**
 * @brief Update screen size after the window has been resized
 * 
 * @param sizex horizontal size
 * @param sizey vertical size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_viewReshape(int sizex, int sizey);

/**
 * @brief Get CPU info
 * 
 * @param names entry name
 * @param flags entry flag
 * @param numNames number of entries
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getCPUInfo(char ***names, bool **flags, unsigned int *numNames);

/**
 * @brief Get compiler flags of this installation
 * 
 * @param flags compiler flags
 * @param numFlags number of flags
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getCompileFlags(char ***flags, unsigned int *numFlags);

/**
 * @brief Get OpenGL info
 * 
 * @param names 
 * @param values 
 * @param numNames 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getGLInfo(char ***names, char ***values, unsigned int *numNames);

/**
 * @brief Get EGL info
 * 
 * @param info 
 * @param numChars
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) MxCSystem_getEGLInfo(char **info, unsigned int *numChars);

#endif // _WRAPS_C_MXCSYSTEM_H_