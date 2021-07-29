%{
    #include "state/MxSpecies.h"

%}

%include "MxSpecies.h"

%extend MxSpecies {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        @property
        def id(self) -> str:
            return self.getId()

        @id.setter
        def id(self, sid: str):
            self.setId(sid)

        @property
        def name(self) -> str:
            return self.getName()

        @name.setter
        def name(self, name: str):
            self.setName(name)

        @property
        def species_type(self) -> str:
            return self.getSpeciesType()

        @species_type.setter
        def species_type(self, sid: str):
            self.setSpeciesType(sid)

        @property
        def compartment(self) -> str:
            return self.getCompartment()

        @compartment.setter
        def compartment(self, sid: str):
            self.setCompartment(sid)

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
        def substance_units(self) -> str:
            return self.getSubstanceUnits()

        @substance_units.setter
        def substance_units(self, sid: str):
            self.setSubstanceUnits(sid)

        @property
        def spatial_size_units(self) -> str:
            return self.getSpatialSizeUnits()

        @spatial_size_units.setter
        def spatial_size_units(self, sid: str):
            self.setSpatialSizeUnits(sid)

        @property
        def units(self) -> str:
            return self.getUnits()

        @units.setter
        def units(self, sname: str):
            self.setUnits(sname)

        @property
        def has_only_substance_units(self) -> bool:
            return self.getHasOnlySubstanceUnits()

        @has_only_substance_units.setter
        def has_only_substance_units(self, value: bool):
            self.setHasOnlySubstanceUnits(value);

        @property
        def boundary_condition(self) -> bool:
            return self.getBoundaryCondition()

        @boundary_condition.setter
        def boundary_condition(self, value: bool):
            self.setBoundaryCondition(value)

        @property
        def charge(self) -> int:
            return self.getCharge()

        @charge.setter
        def charge(self, value: int):
            self.setCharge(value)

        @property
        def constant(self) -> bool:
            return self.getConstant()

        @constant.setter
        def constant(self, value: bool):
            self.setConstant(value)

        @property
        def conversion_factor(self) -> str:
            return self.getConversionFactor()

        @conversion_factor.setter
        def conversion_factor(self, sid: str):
            self.setConversionFactor(sid)
    %}
}

%pythoncode %{
    Species = MxSpecies
%}
