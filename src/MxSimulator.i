%{
    #include "MxSimulator.h"

%}

%ignore MxSimulator::getWindow;

%include "MxSimulator.h"

%extend MxSimulatorPy {
    %pythoncode %{
        @property
        def threads(self) -> int:
            return MxSimulatorPy.getNumThreads()

        @property
        def window(self):
            return MxSimulatorPy.getWindowPy()

        @staticmethod
        def run(*args, **kwargs):
            return MxSimulatorPy._run(args, kwargs)

        @staticmethod
        def show():
            return MxSimulatorPy._show()
    %}
}

%pythoncode %{
    FORWARD_EULER = MxSimulator_EngineIntegrator_FORWARD_EULER
    RUNGE_KUTTA_4 = MxSimulator_EngineIntegrator_RUNGE_KUTTA_4

    Simulator = MxSimulatorPy
%}
