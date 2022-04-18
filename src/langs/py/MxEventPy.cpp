/**
 * @file MxEventPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxEvent
 * @date 2022-03-23
 * 
 */

#include "MxEventPy.h"

#include <MxLogger.h>
#include <MxUniverse.h>


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
    MxUniverse::get()->events->addEvent(event);
    return event;
}
