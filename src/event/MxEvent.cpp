/**
 * @file MxEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic event
 * @date 2021-06-23
 * 
 */

#include "MxEvent.h"

#include <MxPy.h>
#include <MxLogger.h>
#include <engine.h>

MxEvent::MxEvent() : 
    MxEventBase(), 
    invokeMethod(NULL), 
    predicateMethod(NULL) 
{}

MxEvent::MxEvent(MxEventMethod *invokeMethod, MxEventMethod *predicateMethod) : 
    MxEventBase(), 
    invokeMethod(invokeMethod), 
    predicateMethod(predicateMethod) 
{}

MxEvent::~MxEvent() {
    if(invokeMethod) {
        delete invokeMethod;
        invokeMethod = 0;
    }
    if(predicateMethod) {
        delete predicateMethod;
        predicateMethod = 0;
    }
}

HRESULT MxEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}
HRESULT MxEvent::invoke() {
    if(invokeMethod) return (*invokeMethod)(*this);
    return 0;
}

HRESULT MxEvent::eval(const double &time) {
    remove();
    return MxEventBase::eval(time);
}

MxEvent *MxOnEvent(MxEventMethod *invokeMethod, MxEventMethod *predicateMethod) {
    Log(LOG_TRACE);

    MxEvent *event = new MxEvent(invokeMethod, predicateMethod);
    _Engine.events->addEvent(event);
    return event;
}

// python support

MxEventPy::MxEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor) : 
    MxEventBase(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{}

MxEventPy::~MxEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT MxEventPy::predicate() { 
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) return 1;
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT MxEventPy::invoke() {
    if(!invokeExecutor || !invokeExecutor->hasExecutorPyCallable()) return 0;
    invokeExecutor->invoke(*this);
    return invokeExecutor->_result;
}

HRESULT MxEventPy::eval(const double &time) {
    remove();
    return MxEventBase::eval(time);
}


MxEventPy *MxOnEventPy(MxEventPyInvokePyExecutor *invokeExecutor, MxEventPyPredicatePyExecutor *predicateExecutor) {
    Log(LOG_TRACE);

    MxEventPy *event = new MxEventPy(invokeExecutor, predicateExecutor);
    _Engine.events->addEvent(event);
    return event;
}
