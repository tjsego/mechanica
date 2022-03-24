/**
 * @file MxParticleEventSinglePy.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleEventSingle
 * @date 2022-03-23
 * 
 */

#include "MxParticleEventSinglePy.h"

#include <MxLogger.h>
#include <MxUniverse.h>


MxParticleEventSinglePy::MxParticleEventSinglePy(MxParticleType *targetType, 
                                                 MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                 MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                                 MxParticleEventParticleSelector *particleSelector) : 
    MxParticleEventPy(targetType, invokeExecutor, predicateExecutor, particleSelector)
{}

HRESULT MxParticleEventSinglePy::eval(const double &time) {
    remove();
    return MxParticleEventPy::eval(time);
}

MxParticleEventSinglePy *MxOnParticleEventSinglePy(MxParticleType *targetType, 
                                                   MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                   MxParticleEventPyPredicatePyExecutor *predicateExecutor)
{
    Log(LOG_TRACE) << targetType->id;

    MxParticleEventSinglePy *event = new MxParticleEventSinglePy(targetType, invokeExecutor, predicateExecutor);

    MxUniverse::get()->events->addEvent(event);

    return event;
}
