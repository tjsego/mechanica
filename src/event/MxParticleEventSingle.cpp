/**
 * @file MxParticleEventSingle.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic particle single event
 * @date 2021-08-19
 * 
 */
#include "MxParticleEventSingle.h"

#include <MxLogger.h>
#include <MxUniverse.h>

MxParticleEventSingle::MxParticleEventSingle(MxParticleType *targetType, 
                                             MxParticleEventMethod *invokeMethod, 
                                             MxParticleEventMethod *predicateMethod, 
                                             MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEvent(targetType, invokeMethod, predicateMethod, particleSelector) 
{}

HRESULT MxParticleEventSingle::eval(const double &time) {
    remove();
    return MxParticleEvent::eval(time);
}

MxParticleEventSingle *MxOnParticleEventSingle(MxParticleType *targetType, 
                                               MxParticleEventMethod *invokeMethod, 
                                               MxParticleEventMethod *predicateMethod)
{
    MxParticleEventSingle *event = new MxParticleEventSingle(targetType, invokeMethod, predicateMethod);

    MxUniverse::get()->events->addEvent(event);

    return event;
}

// python support

MxParticleSingleEventPy::MxParticleSingleEventPy(MxParticleType *targetType, 
                                                 MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                 MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                                 MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector)
{}

HRESULT MxParticleSingleEventPy::eval(const double &time) {
    remove();
    return MxParticleEventPy::eval(time);
}

MxParticleSingleEventPy *MxOnParticleEventSinglePy(MxParticleType *targetType, 
                                                   MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                   MxParticleEventPyPredicatePyExecutor *predicateExecutor)
{
    Log(LOG_TRACE) << targetType->id;

    MxParticleSingleEventPy *event = new MxParticleSingleEventPy(targetType, invokeExecutor, predicateExecutor);

    MxUniverse::get()->events->addEvent(event);

    return event;
}
