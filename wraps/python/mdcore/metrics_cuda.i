%{
    #include "metrics_cuda.h"

%}

%ignore MxCUDALoadMetrics;
%ignore MxCUDAUnloadMetrics;
%ignore MxCUDAMetrics::load;
%rename(load) MxCUDAMetrics::loadi(const unsigned int&);

%include "metrics_cuda.h"

%pythoncode %{
    CUDAMetrics = MxCUDAMetrics
%}
