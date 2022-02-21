%{
    #include <event/MxEventPyExecutor.h>
%}

%include "MxEventPyExecutor.h"

/**
* Factory for MxEventPyExecutor specializations. 
* Whatever interface is presented to the user should direct 
* the passed python function callback through `init...`, 
* which will set the executor callback in the C++ layer and store 
* the event callback in the python layer for execution. 
* 
* If a specialization defines a class member '_result', then it 
* will be set to the return value of the callback on execution. 
*/
%define MxEventPyExecutor_extender(wrappedName, eventName)

typedef MxEventPyExecutor<eventName> _ ## wrappedName ## TS;
%template(_ ## wrappedName ## TS) MxEventPyExecutor<eventName>;

%pythoncode %{
    def init ## wrappedName(callback):
        ex = wrappedName()
        
        def executorPyCallable():
            execute ## wrappedName(ex, callback)

        ex.setExecutorPyCallable(executorPyCallable)

        return ex

    def execute ## wrappedName(ex, callback):
        result = callback(ex.getEvent())
        if hasattr(ex, '_result'):
            ex._result = result
%}

%enddef
