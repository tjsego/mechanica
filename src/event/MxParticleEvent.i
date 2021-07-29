%{
    #include <event/MxEvent.h>
    #include <event/MxParticleEvent.h>
%}

MxEventPyExecutor_extender(MxParticleEventPyInvokePyExecutor, MxParticleEventPy)
MxEventPyExecutor_extender(MxParticleEventPyPredicatePyExecutor, MxParticleEventPy)
MxEventPyExecutor_extender(MxParticleTimeEventPyInvokePyExecutor, MxParticleTimeEventPy)
MxEventPyExecutor_extender(MxParticleTimeEventPyPredicatePyExecutor, MxParticleTimeEventPy)

%include "MxParticleEvent.h"

%pythoncode %{
    def on_particle(ptype, invoke_method, predicate_method=None, selector="default", single: bool=False):
        invoke_ex = initMxParticleEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxParticleEventPyPredicatePyExecutor(predicate_method)

        if single:
            return MxOnParticleEventSinglePy(ptype, invoke_ex, predicate_ex)
        return MxOnParticleEventPy(ptype, invoke_ex, predicate_ex, selector)


    def on_particletime(ptype, period, invoke_method, predicate_method=None, distribution="default", start_time=0.0, end_time=-1.0, selector="default"):
        invoke_ex = initMxParticleTimeEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxParticleTimeEventPyPredicatePyExecutor(predicate_method)

        return MxOnParticleTimeEventPy(ptype, period, invoke_ex, predicate_ex, distribution, start_time, end_time, selector)


    ParticleEvent = MxParticleEventPy
    ParticleTimeEvent = MxParticleTimeEventPy
%}
