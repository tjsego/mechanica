%{
    #include <event/MxEvent.h>
    #include <event/MxTimeEvent.h>
%}

MxEventPyExecutor_extender(MxEventPyPredicatePyExecutor, MxEventPy)
MxEventPyExecutor_extender(MxEventPyInvokePyExecutor, MxEventPy)

%include "MxEvent.h"

%pythoncode %{
    def on_event(invoke_method, predicate_method=None):
        invoke_ex = initMxEventPyInvokePyExecutor(invoke_method)
        
        predicate_ex = None
        if predicate_method is not None:
            predicate_ex = initMxEventPyPredicatePyExecutor(predicate_method)

        return MxOnEventPy(invoke_ex, predicate_ex)

    Event = MxEventPy
%}
