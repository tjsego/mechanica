/**
 * @file MxParticleEventPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleEvent
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXPARTICLEEVENTPY_H_
#define _SRC_LANGS_PY_MXPARTICLEEVENTPY_H_

#include "MxPy.h"

#include "MxEventPyExecutor.h"

#include <event/MxParticleEvent.h>


struct MxParticleEventPy;

struct MxParticleEventPyPredicatePyExecutor : MxEventPyExecutor<MxParticleEventPy> {
    HRESULT _result = 0;
};

struct MxParticleEventPyInvokePyExecutor : MxEventPyExecutor<MxParticleEventPy> {
    HRESULT _result = 0;
};

struct CAPI_EXPORT MxParticleEventPy : MxParticleEvent {

    MxParticleEventPy() {}
    MxParticleEventPy(MxParticleType *targetType, 
                      MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                      MxParticleEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                      MxParticleEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleEventPy();

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

private:
    
    MxParticleEventPyInvokePyExecutor *invokeExecutor;
    MxParticleEventPyPredicatePyExecutor *predicateExecutor;

};

/**
 * @brief Creates a particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param invokeExecutor an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @param selector name of the function that selects the next particle
 * @return MxParticleEventPy* 
 */
CAPI_FUNC(MxParticleEventPy*) MxOnParticleEventPy(MxParticleType *targetType, 
                                                  MxParticleEventPyInvokePyExecutor *invokeExecutor, 
                                                  MxParticleEventPyPredicatePyExecutor *predicateExecutor, 
                                                  const std::string &selector="default");

#endif // _SRC_LANGS_PY_MXPARTICLEEVENTPY_H_