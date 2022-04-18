/**
 * @file MxTimeEventPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxTimeEvent
 * @date 2022-03-23
 * 
 */

#include "MxTimeEventPy.h"

#include <MxLogger.h>
#include <MxUniverse.h>
#include <engine.h>


MxTimeEventPy::~MxTimeEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT defaultMxTimeEventPyPredicateEval(const MxTimeEventPy &e) {
    auto current_time = _Engine.time * _Engine.dt;
    HRESULT result = current_time >= e.next_time && current_time >= e.start_time && current_time <= e.end_time;

    return result;
}

HRESULT MxTimeEventPy::predicate() {
    if(!predicateExecutor) return defaultMxTimeEventPyPredicateEval(*this);
    else if(!predicateExecutor->hasExecutorPyCallable()) return 1;
    return predicateExecutor->invoke(*this);
}

HRESULT MxTimeEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT MxTimeEventPy::eval(const double &time) {
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);
    return result;
}

double MxTimeEventPy::getNextTime(const double &current_time) {
    if(!nextTimeSetter || nextTimeSetter == NULL) return current_time + this->period;
    return (*this->nextTimeSetter)(*(MxTimeEvent*)this, current_time);
}

MxTimeEventPy* MxOnTimeEventPy(const double &period, 
                               MxTimeEventPyInvokePyExecutor *invokeExecutor, 
                               MxTimeEventPyPredicatePyExecutor *predicateExecutor, 
                               const std::string &distribution, 
                               const double &start_time, 
                               const double &end_time) {
    Log(LOG_TRACE);

    auto itr = MxTimeEventNextTimeSetterNameMap.find(distribution);
    if(itr == MxTimeEventNextTimeSetterNameMap.end()) {
        mx_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    MxTimeEventTimeSetterEnum nextTimeSetterEnum = itr->second;

    MxTimeEventNextTimeSetter *nextTimeSetter = getMxTimeEventNextTimeSetter((MxTimeEventTimeSetterEnum)nextTimeSetterEnum);

    auto event = new MxTimeEventPy(period, invokeExecutor, predicateExecutor, nextTimeSetter, start_time, end_time);
    MxUniverse::get()->events->addEvent(event);
    return event;
}
