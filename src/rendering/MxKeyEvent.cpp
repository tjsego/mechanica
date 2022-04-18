/*
 * MxKeyEvent.cpp
 *
 *  Created on: Dec 29, 2020
 *      Author: andy
 */

#include "MxKeyEvent.hpp"

#include <Magnum/Platform/GlfwApplication.h>

#include <MxLogger.h>
#include <mx_error.h>
#include <iostream>
#include <unordered_map>

template<typename H, typename T>
using MxKeyEventCbStorage = std::unordered_map<H, T*>;

static MxKeyEventCbStorage<Mx_ssize_t, MxKeyEventDelegateType> delegates = {};
static MxKeyEventCbStorage<Mx_ssize_t, MxKeyEventHandlerType> handlers = {};

HRESULT MxKeyEvent::invoke() {
    Log(LOG_TRACE);

    HRESULT r, result = S_OK;
    for(auto &d : delegates) {
        auto f = d.second;
        r = (*f)(glfw_event);
        if(result == S_OK) 
            result = r;
    }
    for(auto &h : handlers) {
        auto f = h.second;
        r = (*f)(this);
        if(result == S_OK) 
            result = r; 
    }

    // TODO: check result code
    return result;
}

HRESULT MxKeyEvent::invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke) {
    auto event = new MxKeyEvent();
    event->glfw_event = &ke;
    auto result = event->invoke();
    delete event;
    return result;
}

template<typename H, typename T>
H addKeyEventCbStorage(MxKeyEventCbStorage<H, T> &storage, T *cb) {
    H i = storage.size();
    for(H j = 0; j < storage.size(); j++) {
        if(storage.find(j) == storage.end()) {
            i = j;
            break;
        }
    }

    storage.insert(std::make_pair(i, cb));
    return i;
}

template<typename H, typename T> 
T *getKeyEventCbStorage(const MxKeyEventCbStorage<H, T> &storage, const H &handle) {
    auto itr = storage.find(handle);
    if(itr == storage.end()) 
        return NULL;
    return itr->second;
}

template<typename H, typename T> 
bool removeKeyEventCbStorage(MxKeyEventCbStorage<H, T> &storage, const H &handle) {
    auto itr = storage.find(handle);
    if(itr == storage.end()) 
        return false;
    storage.erase(itr);
    return true;
}

MxKeyEventDelegateHandle MxKeyEvent::addDelegate(MxKeyEventDelegateType *_delegate) {
    return addKeyEventCbStorage(delegates, _delegate);
}

MxKeyEventHandlerHandle MxKeyEvent::addHandler(MxKeyEventHandlerType *_handler) {
    return addKeyEventCbStorage(handlers, _handler);
}

MxKeyEventDelegateType *MxKeyEvent::getDelegate(const MxKeyEventDelegateHandle &handle) {
    return getKeyEventCbStorage(delegates, handle);
}

MxKeyEventHandlerType *MxKeyEvent::getHandler(const MxKeyEventHandlerHandle &handle) {
    return getKeyEventCbStorage(handlers, handle);
}

bool MxKeyEvent::removeDelegate(const MxKeyEventDelegateHandle &handle) {
    return removeKeyEventCbStorage(delegates, handle);
}

bool MxKeyEvent::removeHandler(const MxKeyEventHandlerHandle &handle) {
    return removeKeyEventCbStorage(handlers, handle);
}

std::string MxKeyEvent::keyName() {
    if(glfw_event) return glfw_event->keyName();
    return "";
}

bool MxKeyEvent::keyAlt() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Alt);
    return false;
}

bool MxKeyEvent::keyCtrl() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Ctrl);
    return false;
}

bool MxKeyEvent::keyShift() {
    if(glfw_event) return bool(glfw_event->modifiers() & Magnum::Platform::GlfwApplication::KeyEvent::Modifier::Shift);
    return false;
}
