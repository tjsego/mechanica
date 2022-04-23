/**
 * @file MxParticleEventPy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleEvent
 * @date 2022-03-23
 * 
 */

#include "MxParticleEventPy.h"

#include <MxLogger.h>
#include <MxUniverse.h>


MxParticleEventPy::MxParticleEventPy(MxParticleType *targetType, 
                                     MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                     MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                     MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEvent(), 
    invokeExecutor(invokeExecutor), 
    predicateExecutor(predicateExecutor)
{
    this->targetType = targetType;
    if (particleSelector==NULL) setMxParticleEventParticleSelector(MxParticleEventParticleSelectorEnum::DEFAULT);
    else setMxParticleEventParticleSelector(particleSelector);
}

MxParticleEventPy::~MxParticleEventPy() {
    if(invokeExecutor) {
        delete invokeExecutor;
        invokeExecutor = 0;
    }
    if(predicateExecutor) {
        delete predicateExecutor;
        predicateExecutor = 0;
    }
}

HRESULT MxParticleEventPy::predicate() {
    if(!predicateExecutor || !predicateExecutor->hasExecutorPyCallable()) return 1;
    predicateExecutor->invoke(*this);
    return predicateExecutor->_result;
}

HRESULT MxParticleEventPy::invoke() {
    if(invokeExecutor && invokeExecutor->hasExecutorPyCallable()) invokeExecutor->invoke(*this);
    return 0;
}

HRESULT MxParticleEventPy::eval(const double &time) {
    targetParticle = getNextParticle();
    return MxEventBase::eval(time);
}

MxParticleEventPy *MxOnParticleEventPy(MxParticleType *targetType, 
                                       MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                       MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                       const std::string &selector)
{
    Log(LOG_TRACE) << targetType->id;

    MxParticleEventParticleSelector *particleSelector = getMxParticleEventParticleSelectorN(selector);
    if (!particleSelector) return NULL;

    MxParticleEventPy *event = new MxParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector);

    MxUniverse::get()->events->addEvent(event);

    return event;
}
