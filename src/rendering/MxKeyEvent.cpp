/*
 * MxKeyEvent.cpp
 *
 *  Created on: Dec 29, 2020
 *      Author: andy
 */

#include "MxKeyEvent.hpp"

#include <MxPy.h>
#include <MxLogger.h>
#include <mx_error.h>
#include <iostream>

static MxKeyEventDelegateType *delegate = NULL;

HRESULT MxKeyEvent::invoke() {
    Log(LOG_TRACE);

    HRESULT result;
    if(delegate) result = (*delegate)(glfw_event);
    else if(MxKeyEventPyExecutor::hasStaticMxKeyEventPyExecutor()) result = MxKeyEventPyExecutor::getStaticMxKeyEventPyExecutor()->invoke();
    else result = S_OK;

    // TODO: check result code
    return result;
}

HRESULT MxKeyEvent::invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke) {
    if(delegate) {
        auto event = new MxKeyEvent();
        event->glfw_event = &ke;
        auto result = event->invoke();
        delete event;
        return result;
    }
    else if(MxKeyEventPyExecutor::hasStaticMxKeyEventPyExecutor()) {
        auto event = new MxKeyEvent();
        event->glfw_event = &ke;
        return MxKeyEventPyExecutor::getStaticMxKeyEventPyExecutor()->invoke(*event);
    }
    return S_OK;
}

HRESULT MxKeyEvent::addDelegate(MxKeyEventDelegateType *_delegate) {
    delegate = _delegate;
    return S_OK;
}

std::string MxKeyEvent::keyName() {
    if(glfw_event) return glfw_event->keyName();
    return "";
}

// python support

static MxKeyEventPyExecutor *staticMxKeyEventPyExecutor = NULL;

bool MxKeyEventPyExecutor::hasStaticMxKeyEventPyExecutor() {
    return staticMxKeyEventPyExecutor != NULL;
}

void MxKeyEventPyExecutor::setStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor) {
    staticMxKeyEventPyExecutor = executor;
}

void MxKeyEventPyExecutor::maybeSetStaticMxKeyEventPyExecutor(MxKeyEventPyExecutor *executor) {
    if(!hasStaticMxKeyEventPyExecutor()) staticMxKeyEventPyExecutor = executor;
}

MxKeyEventPyExecutor *MxKeyEventPyExecutor::getStaticMxKeyEventPyExecutor() {
    return staticMxKeyEventPyExecutor;
}
