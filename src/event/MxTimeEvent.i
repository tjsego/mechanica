%{
    #include <event/MxTimeEvent.h>
%}

MxEventPyExecutor_extender(MxTimeEventPyPredicatePyExecutor, MxTimeEventPy)
MxEventPyExecutor_extender(MxTimeEventPyInvokePyExecutor, MxTimeEventPy)

%include "MxTimeEvent.h"

%pythoncode %{
    def on_time(period, invoke_method, predicate_method=None, distribution="default", start_time=0.0, end_time=-1.0):
        invoke_ex = initMxTimeEventPyInvokePyExecutor(invoke_method)

        if predicate_method is not None:
            predicate_ex = initMxTimeEventPyPredicatePyExecutor(predicate_method)
        else:
            predicate_ex = None

        return MxOnTimeEventPy(period, invoke_ex, predicate_ex, distribution, start_time, end_time)


    TimeEvent = MxTimeEventPy
%}
