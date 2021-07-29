%{
    #include "MxLogger.h"

%}

%include "MxLogger.h"

%extend MxLogger {
    %pythoncode %{
        CURRENT = LOG_CURRENT
        FATAL = LOG_FATAL
        CRITICAL = LOG_CRITICAL
        ERROR = LOG_ERROR
        WARNING = LOG_WARNING
        NOTICE = LOG_NOTICE
        INFORMATION = LOG_INFORMATION
        DEBUG = LOG_DEBUG
        TRACE = LOG_TRACE
    %}
}

%pythoncode %{
    Logger = MxLogger
%}
