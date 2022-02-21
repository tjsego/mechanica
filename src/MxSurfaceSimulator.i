%{
    #include "MxSurfaceSimulator.h"

%}

// problematic for wrapping
%ignore MxSurfaceSimulator::renderBuffer;
%ignore MxSurfaceSimulator::frameBuffer;

// currently not implemented
%ignore MxSurfaceSimulator::step;
%ignore MxSurfaceSimulator::mouseMove;
%ignore MxSurfaceSimulator::mouseClick;

%include "MxSurfaceSimulator.h"

%pythoncode %{
    SurfaceSimulator = MxSurfaceSimulator
    SurfaceSimulatorPy = MxSurfaceSimulatorPy
%}
