%{
    #include <event/MxEvent.h>
    #include <event/MxParticleEvent.h>
%}

MxEventPyExecutor_extender(MxParticleEventPyInvokePyExecutor, MxParticleEventPy)
MxEventPyExecutor_extender(MxParticleEventPyPredicatePyExecutor, MxParticleEventPy)

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


    ParticleEvent = MxParticleEventPy
%}
