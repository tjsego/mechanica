%{
    #include <event/MxParticleTimeEvent.h>
%}

MxEventPyExecutor_extender(MxParticleTimeEventPyInvokePyExecutor, MxParticleTimeEventPy)
MxEventPyExecutor_extender(MxParticleTimeEventPyPredicatePyExecutor, MxParticleTimeEventPy)

%include "MxParticleTimeEvent.h"

%pythoncode %{
    def on_particletime(ptype, period, invoke_method, predicate_method=None, distribution="default", start_time=0.0, end_time=-1.0, selector="default"):
        invoke_ex = initMxParticleTimeEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxParticleTimeEventPyPredicatePyExecutor(predicate_method)

        return MxOnParticleTimeEventPy(ptype, period, invoke_ex, predicate_ex, distribution, start_time, end_time, selector)


    ParticleTimeEvent = MxParticleTimeEventPy
%}
