%{
    #include <event/MxParticleTimeEvent.h>
%}

MxEventPyExecutor_extender(MxParticleTimeEventPyInvokePyExecutor, MxParticleTimeEventPy)
MxEventPyExecutor_extender(MxParticleTimeEventPyPredicatePyExecutor, MxParticleTimeEventPy)

%include "MxParticleTimeEvent.h"

%pythoncode %{
    def on_particletime(ptype, period, invoke_method, predicate_method=None, distribution="default", start_time=0.0, end_time=-1.0, selector="default"):
        """
        A combination of :meth:`on_time` and :meth:`on_particle`. 

        :type ptype: MxParticleType
        :param ptype: the type of particle to select
        :type period: float
        :param period: period of event
        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`MxTimeEvent` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`MxTimeEvent` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        :type distribution: str
        :param distribution: distribution by which the next event time is selected
        :type start_time: str
        :param start_time: time after which the event can occur
        :type end_time: str
        :param end_time: time before which the event can occur; a negative value is interpreted as until 'forever'
        :type selector: str
        :param selector: name of particle selector
        """
        invoke_ex = initMxParticleTimeEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxParticleTimeEventPyPredicatePyExecutor(predicate_method)

        return MxOnParticleTimeEventPy(ptype, period, invoke_ex, predicate_ex, distribution, start_time, end_time, selector)


    ParticleTimeEvent = MxParticleTimeEventPy
%}
