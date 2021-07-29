%{
    #include "MxForce.h"

%}

%include "MxForce.h"

%extend MxConstantForce {
    %pythoncode %{
        @property
        def value(self):
            return self.getValue()

        @value.setter
        def value(self, value):
            self.setValue(value)

        @property
        def period(self):
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
