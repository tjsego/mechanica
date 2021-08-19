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
#include <event/MxEventPyExecutor.h>

struct KeyEvent;

using MxKeyEventDelegateType = HRESULT (*)(Magnum::Platform::GlfwApplication::KeyEvent*);

struct CAPI_EXPORT MxKeyEvent
{
    Magnum::Platform::GlfwApplication::KeyEvent *glfw_event;

    MxKeyEvent(Magnum::Platform::GlfwApplication::KeyEvent *glfw_event=NULL) : glfw_event(glfw_event) {}

    HRESULT invoke();
    static HRESULT invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke);
    // adds an event handle
    static HRESULT addDelegate(MxKeyEventDelegateType *_delegate);

    std::string keyName();
};

// python support

struct CAPI_EXPORT MxKeyEventPyExecutor : MxEventPyExecutor<MxKeyEvent> {
    static bool hasStaticMxKeyEventPyExecutor();
    static void setStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor);
    static void maybeSetStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor);
    static MxKeyEventPyExecutor *getStaticMxKeyEventPyExecutor();
};


#endif /* SRC_RENDERING_MXKEYEVENT_HPP_ */
