%{
    #include "MxForce.h"

%}

%include "MxForce.h"

%extend MxForce {
    %pythoncode %{
        def __reduce__(self):
            return MxForce.fromString, (self.toString(),)
    %}
}

%extend MxForceSum {
    %pythoncode %{
        def __reduce__(self):
            return MxForceSum_fromStr, (self.toString(),)
    %}
}

%extend Berendsen {
    %pythoncode %{
        def __reduce__(self):
            return Berendsen_fromStr, (self.toString(),)
    %}
}

%extend Gaussian {
    %pythoncode %{
        def __reduce__(self):
            return Gaussian_fromStr, (self.toString(),)
    %}
}

%extend Friction {
    %pythoncode %{
        def __reduce__(self):
            return Friction_fromStr, (self.toString(),)
    %}
}

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
    ForceSum = MxForceSum
%}
