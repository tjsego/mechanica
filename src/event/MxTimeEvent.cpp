/**
 * @file MxTimeEvent.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines time-dependent event
 * @date 2021-06-23
 * 
 */

#include "MxTimeEvent.h"
#include <MxUtil.h>
#include <MxLogger.h>
#include <engine.h>
#include <MxPy.h>


HRESULT defaultMxTimeEventPredicateEval(const double &next_time, const double &start_time, const double &end_time) {
    auto current_time = _Engine.time * _Engine.dt;
    HRESULT result = current_time >= next_time;
    if (start_time > 0) result = result && current_time >= start_time;
    if (end_time > 0) result = result && current_time <= end_time;
    return result;
}

MxTimeEvent::~MxTimeEvent() {}

double MxTimeEventSetNextTimeExponential(MxTimeEvent &event, const double &time) {
    std::exponential_distribution<> d(1/event.period);
    return time + d(MxRandom);
}

double MxTimeEventSetNextTimeDeterministic(MxTimeEvent &event, const double &time) {
    return time + event.period;
}

MxTimeEventNextTimeSetter* getMxTimeEventNextTimeSetter(MxTimeEventTimeSetterEnum setterEnum) {
    auto x = MxTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == MxTimeEventNextTimeSetterMap.end()) return NULL;
    return &x->second;
}

MxTimeEventNextTimeSetter* getMxTimeEventNextTimeSetterN(std::string setterName) {
    auto itr = MxTimeEventNextTimeSetterNameMap.find(setterName);
    if(itr == MxTimeEventNextTimeSetterNameMap.end()) {
        mx_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    return getMxTimeEventNextTimeSetter(itr->second);
}

HRESULT MxTimeEvent::predicate() { 
    if(predicateMethod) return (*predicateMethod)(*this);
    return 1;
}

HRESULT MxTimeEvent::invoke() {
    if(invokeMethod) (*invokeMethod)(*this);
    return 0;
}

HRESULT MxTimeEvent::eval(const double &time) {
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);
    return result;
}

double MxTimeEvent::getNextTime(const double &current_time) {
    return (*nextTimeSetter)(*this, current_time);
}

HRESULT MxTimeEvent::setMxTimeEventNextTimeSetter(MxTimeEventTimeSetterEnum setterEnum) {
    auto x = MxTimeEventNextTimeSetterMap.find(setterEnum);
    if (x == MxTimeEventNextTimeSetterMap.end()) return 1;
    this->nextTimeSetter = &x->second;
    return S_OK;
}

MxTimeEvent* MxOnTimeEvent(const double &period, MxTimeEventMethod *invokeMethod, MxTimeEventMethod *predicateMethod, 
    const unsigned int &nextTimeSetterEnum, const double &start_time, const double &end_time) 
{
    Log(LOG_TRACE);

    MxTimeEventNextTimeSetter *nextTimeSetter = getMxTimeEventNextTimeSetter((MxTimeEventTimeSetterEnum)nextTimeSetterEnum);
    
    MxTimeEvent *event = new MxTimeEvent(period, invokeMethod, predicateMethod, nextTimeSetter, start_time, end_time);
    
    _Engine.events->addEvent(event);

    return event;

}

MxTimeEvent* MxOnTimeEventN(const double &period, 
                            MxTimeEventMethod *invokeMethod, 
                            MxTimeEventMethod *predicateMethod, 
                            const std::string &distribution, 
                            const double &start_time, 
                            const double &end_time) 
{
    auto itr = MxTimeEventNextTimeSetterNameMap.find(distribution);
    if(itr == MxTimeEventNextTimeSetterNameMap.end()) {
        mx_error(E_FAIL, "Invalid distribution");
        return NULL;
    }
    MxTimeEventTimeSetterEnum nextTimeSetterEnum = itr->second;

    return MxOnTimeEvent(period, invokeMethod, predicateMethod, (unsigned)nextTimeSetterEnum, start_time, end_time);
}

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
    _Engine.events->addEvent(event);
    return event;
}
