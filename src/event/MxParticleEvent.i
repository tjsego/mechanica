%{
    #include <event/MxEvent.h>
    #include <event/MxParticleEvent.h>
    #include <event/MxParticleEventSingle.h>
%}

MxEventPyExecutor_extender(MxParticleEventPyInvokePyExecutor, MxParticleEventPy)
MxEventPyExecutor_extender(MxParticleEventPyPredicatePyExecutor, MxParticleEventPy)

%include "MxParticleEvent.h"
%include "MxParticleEventSingle.h"

%pythoncode %{
    def on_particle(ptype, invoke_method, predicate_method=None, selector="default", single: bool=False):
        """
        Like :meth:`on_event`, but for a particle of a particular particle type. 

        :type ptype: MxParticleType
        :param ptype: the type of particle to select
        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`MxParticleEvent` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`MxParticleEvent` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        :type selector: str
        :param selector: name of particle selector
        :type single: bool
        :param single: flag to only trigger event once and then return
        """
        invoke_ex = initMxParticleEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxParticleEventPyPredicatePyExecutor(predicate_method)

        if single:
            return MxOnParticleEventSinglePy(ptype, invoke_ex, predicate_ex)
        return MxOnParticleEventPy(ptype, invoke_ex, predicate_ex, selector)


    ParticleEvent = MxParticleEventPy
%}
