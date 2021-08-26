%{
    #include <event/MxEvent.h>
    #include <event/MxTimeEvent.h>
%}

MxEventPyExecutor_extender(MxEventPyPredicatePyExecutor, MxEventPy)
MxEventPyExecutor_extender(MxEventPyInvokePyExecutor, MxEventPy)

%include "MxEvent.h"

%pythoncode %{
    def on_event(invoke_method, predicate_method=None):
        """
        Creates and registers an event using prescribed invoke and predicate python function executors

        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`MxEvent` instance as argument and returns 0 on success, 1 on error
        :type invoke_method: PyObject
        :param predicate_method: optional predicate method; decides if an event occurs. 
            Takes a :class:`MxEvent` instance as argument and returns 1 to trigger event, -1 on error and 0 otherwise
        """
        invoke_ex = initMxEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxEventPyPredicatePyExecutor(predicate_method)

        return MxOnEventPy(invoke_ex, predicate_ex)

    Event = MxEventPy
%}
