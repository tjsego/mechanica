%{
    #include "MxSystem.h"

%}

%ignore MxSystem::cpuInfo;
%ignore MxSystem::compileFlags;
%ignore MxSystem::glInfo;
%ignore MxSystem::eglInfo;

%include "MxSystem.h"

%pythoncode %{
    system = getSystemPy()
%}
