/**
 * @file MxParticleEventSingle.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines basic particle single event
 * @date 2021-08-19
 * 
 */
#ifndef SRC_EVENT_MXPARTICLEEVENTSINGLE_H_
#define SRC_EVENT_MXPARTICLEEVENTSINGLE_H_

#include "MxParticleEvent.h"

// Single particle event
struct CAPI_EXPORT MxParticleEventSingle : MxParticleEvent {

    MxParticleEventSingle(MxParticleType *targetType, 
                          MxParticleEventMethod *invokeMethod, 
                          MxParticleEventMethod *predicateMethod, 
                          MxParticleEventParticleSelector *particleSelector=NULL);

    virtual HRESULT eval(const double &time);

};

/**
 * @brief Creates a single particle event using prescribed invoke and predicate functions
 * 
 * @param targetType target particle type
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return MxParticleEventSingle* 
 */
CAPI_FUNC(MxParticleEventSingle*) MxOnParticleEventSingle(MxParticleType *targetType, 
                                                          MxParticleEventMethod *invokeMethod, 
                                                          MxParticleEventMethod *predicateMethod);

#endif // SRC_EVENT_MXPARTICLEEVENTSINGLE_H_