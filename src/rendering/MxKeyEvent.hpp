/*
 * MxKeyEvent.h
 *
 *  Created on: Dec 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXKEYEVENT_HPP_
#define SRC_RENDERING_MXKEYEVENT_HPP_

#include <Magnum/Platform/GlfwApplication.h>
#include <mx_port.h>

struct KeyEvent;

using MxKeyEventDelegateType = HRESULT (*)(Magnum::Platform::GlfwApplication::KeyEvent*);
using MxKeyEventHandlerType = HRESULT (*)(struct MxKeyEvent*);

typedef Mx_ssize_t MxKeyEventDelegateHandle;
typedef Mx_ssize_t MxKeyEventHandlerHandle;

struct CAPI_EXPORT MxKeyEvent
{
    Magnum::Platform::GlfwApplication::KeyEvent *glfw_event;

    MxKeyEvent(Magnum::Platform::GlfwApplication::KeyEvent *glfw_event=NULL) : glfw_event(glfw_event) {}

    HRESULT invoke();
    static HRESULT invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke);
    
    /**
     * @brief Adds an event delegate
     * 
     * @param _delegate delegate to add
     * @return handle for future getting and removing 
     */
    static MxKeyEventDelegateHandle addDelegate(MxKeyEventDelegateType *_delegate);

    /**
     * @brief Adds an event handler
     * 
     * @param _handler handler to add
     * @return handle for future getting and removing
     */
    static MxKeyEventHandlerHandle addHandler(MxKeyEventHandlerType *_handler);

    /**
     * @brief Get an event delegate
     * 
     * @param handle delegate handle
     * @return delegate if handle is valid, otherwise NULL
     */
    static MxKeyEventDelegateType *getDelegate(const MxKeyEventDelegateHandle &handle);

    /**
     * @brief Get an event handler
     * 
     * @param handle handler handle
     * @return handler if handle is valid, otherwise NULL
     */
    static MxKeyEventHandlerType *getHandler(const MxKeyEventHandlerHandle &handle);

    /**
     * @brief Remove an event delegate
     * 
     * @param handle delegate handle
     * @return true when delegate is removed
     * @return false when handle is invalid
     */
    static bool removeDelegate(const MxKeyEventDelegateHandle &handle);

    /**
     * @brief Remove an event handler
     * 
     * @param handle handler handle
     * @return true when handler is removed
     * @return false when handle is invalid
     */
    static bool removeHandler(const MxKeyEventHandlerHandle &handle);

    std::string keyName();
    bool keyAlt();
    bool keyCtrl();
    bool keyShift();
};

#endif /* SRC_RENDERING_MXKEYEVENT_HPP_ */
