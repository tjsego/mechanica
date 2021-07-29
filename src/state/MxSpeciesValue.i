%{
    #include "state/MxSpeciesValue.h"

%}

%include "MxSpeciesValue.h"

%extend MxSpeciesValue {
    %pythoncode %{
        @property
        def boundary_condition(self) -> bool:
            return self.getBoundaryCondition()

        def initial_amount(self) -> float:
            return self.getInitialAmount()

        def initial_concentration(self) -> float:
            return self.getInitialConcentration()

        def constant(self) -> bool:
            return self.getConstant()
    %}
}

%pythoncode %{
    SpeciesValue = MxSpeciesValue
%}
