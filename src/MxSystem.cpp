/*
 * MxSystem.cpp
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#include <MxSystem.h>
#include <MxSimulator.h>
#include <rendering/MxWindowlessApplication.h>
#include <rendering/MxWindowless.h>
#include <rendering/MxApplication.h>
#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxGlfwApplication.h>
#include <rendering/MxClipPlane.hpp>
#include <MxLogger.h>
#include <mx_error.h>
#include <sstream>

static double ms(ticks tks)
{
    return (double)tks / (_Engine.time * CLOCKS_PER_SEC);
}

std::tuple<char*, size_t> MxSystem::testImage() {
    return MxTestImage();
}

std::tuple<char*, size_t> MxSystem::imageData() {
    return MxFramebufferImageData();
}

bool MxSystem::contextHasCurrent() {
    try {
        std::thread::id id = std::this_thread::get_id();
        Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        MxSimulator *sim = MxSimulator::get();
        
        return sim->app->contextHasCurrent();
        
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::contextMakeCurrent() {
    try {
        std::thread::id id = std::this_thread::get_id();
        Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        MxSimulator *sim = MxSimulator::get();
        sim->app->contextMakeCurrent();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::contextRelease() {
    try {
        std::thread::id id = std::this_thread::get_id();
        Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        MxSimulator *sim = MxSimulator::get();
        sim->app->contextRelease();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraMoveTo(const MxVector3f &eye, const MxVector3f &center, const MxVector3f &up) {
    try {
        MxUniverseRenderer *rend = MxSimulator::get()->app->getRenderer();

        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->setViewParameters(eye, center, up);

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraMoveTo(const MxVector3f &center, const MxQuaternionf &rotation, const float &zoom) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->setViewParameters(center, rotation, zoom);

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewBottom() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewBottom(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewTop() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewTop(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewLeft() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewLeft(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewRight() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewRight(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewBack() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewBack(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraViewFront() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewFront(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraReset() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
            
        ab->reset();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraRotateMouse(const MxVector2i &mousePos) {
    try {
        Log(LOG_TRACE);
        
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->rotate(mousePos);
        
        ab->updateTransformation();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraTranslateMouse(const MxVector2i &mousePos) {
    try {
        Log(LOG_TRACE);
        
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->translate(mousePos);
        
        ab->updateTransformation();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraInitMouse(const MxVector2i &mousePos) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->initTransformation(mousePos);
        
        MxSimulator::get()->redraw();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraTranslateBy(const MxVector2f &trans) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->translateDelta(trans);

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraZoomBy(const float &delta) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->zoom(delta);
        
        ab->updateTransformation();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraZoomTo(const float &distance) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->zoomTo(distance);

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraRotateToAxis(const MxVector3f &axis, const float &distance) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->rotateToAxis(axis, distance);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraRotateToEulerAngle(const MxVector3f &angles) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->rotateToEulerAngles(angles);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::cameraRotateByEulerAngle(const MxVector3f &angles) {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->rotateByEulerAngles(angles);
        
        MxSimulator::get()->redraw();

        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::cameraCenter() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        return ab->cposition();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxQuaternionf MxSystem::cameraRotation() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        return ab->crotation();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

float MxSystem::cameraZoom() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBallCamera *ab = renderer->_arcball;
        
        return ab->czoom();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

struct MxUniverseRenderer *MxSystem::getRenderer() {
    try {
        MxSimulator *sim = MxSimulator::get();
        
        return sim->app->getRenderer();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

MxVector3f MxSystem::getAmbientColor() {
    try {
        return MxSystem::getRenderer()->ambientColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setAmbientColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setAmbientColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setAmbientColor(const MxVector3f &color, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setAmbientColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getDiffuseColor() {
    try {
        return MxSystem::getRenderer()->diffuseColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setDiffuseColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setDiffuseColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setDiffuseColor(const MxVector3f &color, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setDiffuseColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getSpecularColor() {
    try {
        return MxSystem::getRenderer()->specularColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setSpecularColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setSpecularColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setSpecularColor(const MxVector3f &color, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setSpecularColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

float MxSystem::getShininess() {
    try {
        return MxSystem::getRenderer()->shininess();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setShininess(const float &shininess) {
    try {
        MxSystem::getRenderer()->setShininess(shininess);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setShininess(const float &shininess, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setShininess(shininess);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getGridColor() {
    try {
        return MxSystem::getRenderer()->gridColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setGridColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setGridColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getSceneBoxColor() {
    try {
        return MxSystem::getRenderer()->sceneBoxColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setSceneBoxColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setSceneBoxColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getLightDirection() {
    try {
        return MxSystem::getRenderer()->lightDirection();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setLightDirection(const MxVector3f& lightDir) {
    try {
        MxSystem::getRenderer()->setLightDirection(lightDir);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setLightDirection(const MxVector3f& lightDir, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setLightDirection(lightDir);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

MxVector3f MxSystem::getLightColor() {
    try {
        return MxSystem::getRenderer()->lightColor();
    }
    catch(const std::exception &e) {
        MX_RETURN_EXP(e);
    }
}

HRESULT MxSystem::setLightColor(const MxVector3f &color) {
    try {
        MxSystem::getRenderer()->setLightColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::setLightColor(const MxVector3f &color, const unsigned int &srFlag) {
    try {
        MxSystem::getRenderer()->getSubRenderer((MxSubRendererFlag)srFlag)->setLightColor(color);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::decorateScene(const bool &decorate) {
    try {
        MxSystem::getRenderer()->decorateScene(decorate);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

HRESULT MxSystem::viewReshape(const MxVector2i &windowSize) {
    try {
        MxSimulator *sim = MxSimulator::get();

        MxUniverseRenderer *renderer = sim->app->getRenderer();
        
        Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
        ab->reshape(windowSize);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        mx_exp(e);
        return E_FAIL;
    }
}

std::string MxSystem::performanceCounters() {
    std::stringstream ss;
    
    ss << "performance_timers : { " << std::endl;
    ss << "\t name: " << Universe.name << "," << std::endl;
    ss << "\t wall_time: " << MxWallTime() << "," << std::endl;
    ss << "\t cpu_time: " << MxCPUTime() << "," << std::endl;
    ss << "\t fps: " << engine_steps_per_second() << "," << std::endl;
    ss << "\t kinetic energy: " << engine_kinetic_energy(&_Engine) << "," << std::endl;
    ss << "\t step: " << ms(_Engine.timers[engine_timer_step]) << "," << std::endl;
    ss << "\t nonbond: " << ms(_Engine.timers[engine_timer_nonbond]) << "," << std::endl;
    ss << "\t bonded: " << ms(_Engine.timers[engine_timer_bonded]) << "," << std::endl;
    ss << "\t advance: " << ms(_Engine.timers[engine_timer_advance]) << "," << std::endl;
    ss << "\t rendering: " << ms(_Engine.timers[engine_timer_render]) << "," << std::endl;
    ss << "\t total: " << ms(_Engine.timers[engine_timer_render] + _Engine.timers[engine_timer_step]) << "," << std::endl;
    ss << "\t time_steps: " << _Engine.time  << std::endl;
    ss << "}" << std::endl;
    
    return ss.str();
}

std::unordered_map<std::string, bool> MxSystem::cpu_info() {
    return getFeaturesMap();
}

std::list<std::string> MxSystem::compile_flags() {
    return MxCompileFlags().getFlags();
}

std::unordered_map<std::string, std::string> MxSystem::gl_info() {
    return Mx_GlInfo();
}

std::string MxSystem::egl_info() {
    return Mx_EglInfo();
}

std::unordered_map<std::string, std::string> MxSystem::test_headless() {
#if defined(MX_APPLE)
    return Mx_GlInfo();

#elif defined(MX_LINUX)
    return Mx_GlInfo();
}
#elif defined(MX_WINDOWS)
    return Mx_GlInfo();
}
#else
#error no windowless application available on this platform
#endif

PyObject *MxSystemPy::test_image() {
    return MxTestImage(Py_None);
}

PyObject *MxSystemPy::image_data() {
    return MxFramebufferImageData(Py_None);
}

bool MxSystemPy::is_terminal_interactive() {
    return Mx_TerminalInteractiveShell();
}

bool MxSystemPy::is_jupyter_notebook() {
    return Mx_ZMQInteractiveShell();
}

PyObject *MxSystemPy::jwidget_init(PyObject *args, PyObject *kwargs) {
    
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        mx_error(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* init = PyObject_GetAttrString(module,(char*)"init");
    
    if(!init) {
        mx_error(E_FAIL, "mechanica.jwidget package does not have an init function");
        return NULL;
    }

    PyObject* result = PyObject_Call(init, args, kwargs);
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(init);
    
    if(!result) {
        Log(LOG_ERROR) << "error calling mechanica.jwidget.init: " << mx::pyerror_str();
    }
    
    return result;
}

PyObject *MxSystemPy::jwidget_run(PyObject *args, PyObject *kwargs) {
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        mx_error(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* run = PyObject_GetAttrString(module,(char*)"run");
    
    if(!run) {
        mx_error(E_FAIL, "mechanica.jwidget package does not have an run function");
        return NULL;
    }

    PyObject* result = PyObject_Call(run, args, kwargs);

    if (!result) {
        Log(LOG_ERROR) << "error calling mechanica.jwidget.run: " << mx::pyerror_str();
    }

    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(run);
    
    return result;
    
}

void MxPrintPerformanceCounters() {
    MxLoggingBuffer log(LOG_NOTICE, NULL, NULL, -1);
    log.stream() << MxSystem::performanceCounters();
}

static Magnum::Debug *magnum_debug = NULL;
static Magnum::Warning *magnum_warning = NULL;
static Magnum::Error *magnum_error = NULL;

HRESULT MxLoggerCallbackImpl(MxLogEvent, std::ostream *os) {
    Log(LOG_TRACE);
    
    delete magnum_debug; magnum_debug = NULL;
    delete magnum_warning; magnum_warning = NULL;
    delete magnum_error; magnum_error = NULL;
    
    if(MxLogger::getLevel() >= LOG_ERROR) {
        Log(LOG_DEBUG) << "setting Magnum::Error to Mechanica log output";
        magnum_error = new Magnum::Error(os);
    }
    else {
        magnum_error = new Magnum::Error(NULL);
    }
    
    if(MxLogger::getLevel() >= LOG_WARNING) {
        Log(LOG_DEBUG) << "setting Magnum::Warning to Mechanica log output";
        magnum_warning = new Magnum::Warning(os);
    }
    else {
        magnum_warning = new Magnum::Warning(NULL);
    }
    
    if(MxLogger::getLevel() >= LOG_DEBUG) {
        Log(LOG_DEBUG) << "setting Magnum::Debug to Mechanica log output";
        magnum_debug = new Magnum::Debug(os);
    }
    else {
        magnum_debug = new Magnum::Debug(NULL);
    }
    
    return S_OK;
}
