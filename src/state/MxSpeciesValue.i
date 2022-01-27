%{
    #include "state/MxSpeciesValue.h"

%}

%rename(_secrete1) MxSpeciesValue::secrete(const double &, const struct MxParticleList &);
%rename(_secrete2) MxSpeciesValue::secrete(const double &, const double &);

%include "MxSpeciesValue.h"

%extend MxSpeciesValue {
    %pythoncode %{
        def secrete(self, amount, to=None, distance=None):
            """
            Secrete this species into a neighborhood.

            Requires either a list of neighboring particles or neighborhood distance.

            :param amount: Amount to secrete.
            :type amount: float
            :param to: Optional list of particles to secrete to.
            :type to: MxParticleList
            :param distance: Neighborhood distance.
            :type distance: float
            :return: Amount actually secreted, accounting for availability and other subtleties. 
            :rtype: float
            """

            if to is not None:
                return self._secrete1(amount, to)
            elif distance is not None:
                return self._secrete2(amount, distance)
            raise ValueError('A neighbor list or neighbor distance must be specified')

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
