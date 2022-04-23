/**
 * @file MxParticleTimeEventPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleTimeEvent
 * @date 2022-03-23
 * 
 */

#include "MxParticleTimeEventPy.h"

#include <MxLogger.h>
#include <MxUniverse.h>


MxParticleTimeEventPy::MxParticleTimeEventPy(MxParticleType *targetType, 
                                             const double &period, 
                                             MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                             MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
                                             MxParticleTimeEventNextTimeSetter *nextTimeSetter, 
                                             const double &start_time, 
                                             const double &end_time,
                                             MxParticleTimeEventParticleSelector *particleSelector) : 
    MxParticleTimeEvent(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    this->period = period;
    this->next_time = 0;
    this->start_time = start_time;
    this->end_time = end_time > 0 ? end_time : std::numeric_limits<double>::max();

    if (nextTimeSetter == NULL) setMxParticleTimeEventNextTimeSetter(MxParticleTimeEventTimeSetterEnum::DEFAULT);
    else setMxParticleTimeEventNextTimeSetter(nextTimeSetter);

    if (particleSelector == NULL) setMxParticleTimeEventParticleSelector(MxParticleTimeEventParticleSelectorEnum::DEFAULT);
    else setMxParticleTimeEventParticleSelector(particleSelector);
}

MxParticleTimeEventPy::~MxParticleTimeEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT MxParticleTimeEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) {
        return defaultMxTimeEventPredicateEval(this->next_time, this->start_time, this->end_time);
    }
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT MxParticleTimeEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT MxParticleTimeEventPy::eval(const double &time) {
    targetParticle = getNextParticle();
    auto result = MxEventBase::eval(time);
    if(result) this->next_time = getNextTime(time);

    return result;
}

MxParticleTimeEventPy *MxOnParticleTimeEventPy(MxParticleType *targetType, 
                                               const double &period, 
                                               MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                               MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor, 
                                               const std::string &distribution, 
                                               const double &start_time, 
                                               const double &end_time, 
                                               const std::string &selector)
{
    Log(LOG_TRACE) << targetType->id;
    
    MxParticleTimeEventNextTimeSetter *nextTimeSetter = getMxParticleTimeEventNextTimeSetterN(distribution);
    if (!nextTimeSetter) return NULL;

    MxParticleTimeEventParticleSelector *particleSelector = getMxParticleTimeEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;
    
    MxParticleTimeEventPy *event = new MxParticleTimeEventPy(targetType, period, invokeExecutor, predicateExecutor, nextTimeSetter, start_time, end_time, particleSelector);

    MxUniverse::get()->events->addEvent(event);

    return event;
}
