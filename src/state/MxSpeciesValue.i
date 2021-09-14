%{
    #include "state/MxSpeciesValue.h"

%}

%include "MxSpeciesValue.h"

%extend MxSpeciesValue {
    %pythoncode %{
        @property
        def boundary_condition(self) -> bool:
            return self.getBoundaryCondition()

        @boundary_condition.setter
        def boundary_condition(self, value: int):
            self.setBoundaryCondition(value)

        @property
        def initial_amount(self) -> float:
            return self.getInitialAmount()

        @initial_amount.setter
        def initial_amount(self, value: float):
            self.setInitialAmount(value)

        @property
        def initial_concentration(self) -> float:
            return self.getInitialConcentration()

        @initial_concentration.setter
        def initial_concentration(self, value: float):
            self.setInitialConcentration(value)

        @property
        def constant(self) -> bool:
            return self.getConstant()

        @constant.setter
        def constant(self, value: int):
            self.setConstant(value)
    %}
}

%pythoncode %{
    SpeciesValue = MxSpeciesValue
%}
