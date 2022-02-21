%{
    #include "MxLogger.h"

%}

%include "MxLogger.h"

%extend MxLogger {
    %pythoncode %{
        CURRENT = LOG_CURRENT  #: :meta hide-value:
        FATAL = LOG_FATAL  #: :meta hide-value:
        CRITICAL = LOG_CRITICAL  #: :meta hide-value:
        ERROR = LOG_ERROR  #: :meta hide-value:
        WARNING = LOG_WARNING  #: :meta hide-value:
        NOTICE = LOG_NOTICE  #: :meta hide-value:
        INFORMATION = LOG_INFORMATION  #: :meta hide-value:
        DEBUG = LOG_DEBUG  #: :meta hide-value:
        TRACE = LOG_TRACE  #: :meta hide-value:
    %}
}

%pythoncode %{
    Logger = MxLogger
%}
