/**
 * @file MxEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic event
 * @date 2021-06-23
 * 
 */

#include "MxEvent.h"

#include <MxLogger.h>
#include <MxUniverse.h>

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
    MxUniverse::get()->events->addEvent(event);
    return event;
}
