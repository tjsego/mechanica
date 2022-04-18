/**
 * @file MxParticleEventSinglePy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleEventSingle
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXPARTICLEEVENTSINGLEPY_H_
#define _SRC_LANGS_PY_MXPARTICLEEVENTSINGLEPY_H_

#include "MxPy.h"

#include "MxParticleEventPy.h"


// Single particle event
struct CAPI_EXPORT MxParticleEventSinglePy : MxParticleEventPy {
    MxParticleEventSinglePy(MxParticleType *targetType, 
                            MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                            MxParticleEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                            MxParticleEventParticleSelector *particleSelector=NULL);

    virtual HRESULT eval(const double &time);
};

/**
 * @brief Creates a single particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param invokeMethod an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @return MxParticleEventSinglePy* 
 */
CAPI_FUNC(MxParticleEventSinglePy*) MxOnParticleEventSinglePy(MxParticleType *targetType, 
                                                              MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                              MxParticleEventPyPredicatePyExecutor *predicateExecutor);

#endif // _SRC_LANGS_PY_MXPARTICLEEVENTSINGLEPY_H_