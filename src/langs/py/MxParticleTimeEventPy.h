/**
 * @file MxParticleTimeEventPy.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines Python support for MxParticleTimeEvent
 * @date 2022-03-23
 * 
 */
#ifndef _SRC_LANGS_PY_MXPARTICLETIMEEVENTPY_H_
#define _SRC_LANGS_PY_MXPARTICLETIMEEVENTPY_H_

#include "MxPy.h"

#include "MxEventPyExecutor.h"

#include <event/MxParticleTimeEvent.h>


struct MxParticleTimeEventPy;

struct MxParticleTimeEventPyPredicatePyExecutor : MxEventPyExecutor<MxParticleTimeEventPy> {
    HRESULT _result = 0;
};

struct MxParticleTimeEventPyInvokePyExecutor : MxEventPyExecutor<MxParticleTimeEventPy> {
    HRESULT _result = 0;
};

// Time-dependent particle event
struct CAPI_EXPORT MxParticleTimeEventPy : MxParticleTimeEvent {
    
    MxParticleTimeEventPy() {}
    MxParticleTimeEventPy(MxParticleType *targetType, 
                          const double &period, 
                          MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                          MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                          MxParticleTimeEventNextTimeSetter *nextTimeSetter=NULL, 
                          const double &start_time=0, 
                          const double &end_time=-1,
                          MxParticleTimeEventParticleSelector *particleSelector=NULL);
    virtual ~MxParticleTimeEventPy();

    virtual HRESULT predicate();
    virtual HRESULT invoke();
    virtual HRESULT eval(const double &time);

private:

    MxParticleTimeEventPyInvokePyExecutor *invokeExecutor;
    MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor;

};

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate python function executors
 * 
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke python function executor; evaluated when an event occurs
 * @param predicateMethod a predicate python function executor; evaluated to determine if an event occurs
 * @param distribution name of the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param selector name of the function that selects the next particle
 * @return MxParticleTimeEvent* 
 */
CAPI_FUNC(MxParticleTimeEventPy*) MxOnParticleTimeEventPy(MxParticleType *targetType, 
                                                          const double &period, 
                                                          MxParticleTimeEventPyInvokePyExecutor *invokeExecutor, 
                                                          MxParticleTimeEventPyPredicatePyExecutor *predicateExecutor=NULL, 
                                                          const std::string &distribution="default", 
                                                          const double &start_time=0.0, 
                                                          const double &end_time=-1.0, 
                                                          const std::string &selector="default");

#endif // _SRC_LANGS_PY_MXPARTICLETIMEEVENTPY_H_