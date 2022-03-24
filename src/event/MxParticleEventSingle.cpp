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
