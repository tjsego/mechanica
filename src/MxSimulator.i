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
            return MxSimulatorPy.getWindow()

        @staticmethod
        def run(*args, **kwargs):
            """
            Runs the event loop until all windows close or simulation time expires. 
            Automatically performs universe time propogation. 

            :type args: double
            :param args: final time (default runs infinitly)
            """    
            return MxSimulatorPy._run(args, kwargs)

        @staticmethod
        def show():
            """
            Shows any windows that were specified in the config. This works just like
            MatPlotLib's ``show`` method. The ``show`` method does not start the
            universe time propagation unlike ``run`` and ``irun``.
            """
            return MxSimulatorPy._show()
    %}
}

%pythoncode %{
    FORWARD_EULER = MxSimulator_EngineIntegrator_FORWARD_EULER
    RUNGE_KUTTA_4 = MxSimulator_EngineIntegrator_RUNGE_KUTTA_4

    Simulator = MxSimulatorPy
%}
