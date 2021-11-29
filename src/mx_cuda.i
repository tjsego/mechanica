%{
    #include "mx_cuda.h"

    #include "cuda/MxSimulatorCUDAConfig.h"
    #include "cuda/MxBondCUDAConfig.h"
    #include "cuda/MxEngineCUDAConfig.h"

%}

%ignore mx_cuda_errorchk;
%ignore mx_nvrtc_errorchk;
%ignore mx_cudart_errorchk;

%ignore MxCUDARTSource;
%ignore MxCudaRTProgram;
%ignore MxCUDAFunction;
%ignore MxCudaContext;
%ignore MxCUDADevice;

%include "mx_cuda.h"
%include "cuda/MxBondCUDAConfig.h"
%include "cuda/MxEngineCUDAConfig.h"
%include "cuda/MxSimulatorCUDAConfig.h"

%pythoncode %{
    cuda = MxCUDA
%}
