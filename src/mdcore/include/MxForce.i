%{
    #include "MxForce.h"

%}

%include "MxForce.h"

%extend MxConstantForce {
    %pythoncode %{
        @property
        def value(self):
            """
            Current value of the force. 
            
            This can be set to a function that takes no arguments and returns a 3-component list of floats. 
            """
            return self.getValue()

        @value.setter
        def value(self, value):
            self.setValue(value)

        @property
        def period(self):
            """Period of the force"""
            return self.getPeriod()

        @period.setter
        def period(self, period):
            self.setPeriod(period)
    %}
}

%pythoncode %{
    Force = MxForce
    ConstantForce = MxConstantForcePy
%}
