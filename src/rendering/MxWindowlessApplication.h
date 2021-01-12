/*
 * MxWindowlessApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#ifndef SRC_MXWINDOWLESSAPPLICATION_H_
#define SRC_MXWINDOWLESSAPPLICATION_H_

#include <mx_config.h>
#include <Mechanica.h>
#include <rendering/MxApplication.h>
#include <Magnum/GL/Context.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>


#if defined(MX_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(MX_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
#elif defined(MX_WINDOWS)
#include "Magnum/Platform/WindowlessWglApplication.h"
#else
#error no windowless application available on this platform
#endif




struct  MxWindowlessApplication :
        public MxApplication,
        private Magnum::Platform::WindowlessApplication
{
public:

    typedef Magnum::Platform::WindowlessApplication::Arguments Arguments;

    typedef Magnum::Platform::WindowlessApplication::Configuration Configuration;

    MxWindowlessApplication() = delete;

    /**
     * Set up the app, but don't create a context just yet.
     */
    MxWindowlessApplication(const Arguments &args);

    ~MxWindowlessApplication();
    
    MxUniverseRenderer *getRenderer() override;
    
    
    HRESULT createContext(const MxSimulator::Config &conf) override;





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
     HRESULT pollEvents () override;

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
     HRESULT waitEvents () override;

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

    HRESULT waitEventsTimeout(double  timeout) override;


    /**
     * This function posts an empty event from the current thread
     * to the event queue, causing waitEvents or waitEventsTimeout to return.
     */
    HRESULT postEmptyEvent() override;

    HRESULT setSwapInterval(int si) override { return E_NOTIMPL;};
    
    HRESULT mainLoopIteration(double timeout) override;
    
    struct MxGlfwWindow *getWindow() override;
    
    int windowAttribute(MxWindowAttributes attr) override;
    
    HRESULT setWindowAttribute(MxWindowAttributes attr, int val) override;
    
    HRESULT redraw() override;
    
    HRESULT close() override;
    
    HRESULT destroy() override;
    
    HRESULT show() override;
    
    HRESULT messageLoop() override;
    
    Magnum::GL::AbstractFramebuffer& framebuffer() override;
    

private:
    virtual int exec() override { return 0; };

    typedef Magnum::Platform::WindowlessApplication WindowlessApplication;
    
    Magnum::GL::Renderbuffer renderBuffer;
    
    Magnum::GL::Framebuffer frameBuffer;
    

    
    struct MxWindowlessWindow *window;
    struct MxUniverseRenderer *renderer;
    Magnum::Vector2i frameBufferSize;
    
    friend class MxWindowlessWindow;

};

#endif /* SRC_MXWINDOWLESSAPPLICATION_H_ */
