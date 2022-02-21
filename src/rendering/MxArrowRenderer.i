%{
    #include "rendering/MxArrowRenderer.h"
    
%}

%include "MxArrowRenderer.h"

%template(pairInt_MxArrowData_p) std::pair<int, MxArrowData*>;

%pythoncode %{
    ArrowData = MxArrowData
    ArrowRenderer = MxArrowRenderer
%}
