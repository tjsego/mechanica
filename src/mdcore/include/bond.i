%{
    #include "bond.h"

%}

%rename(_getitem) MxBondHandle::operator[];

%include "bond.h"

%template(vectorBondHandle_p) std::vector<MxBondHandle*>;

%extend MxBondHandle {
    %pythoncode %{
        def __getitem__(self, index: int):
            return self._getitem(index)

        def __str__(self) -> str:
            return self.str()

        @property
        def energy(self):
            """bond energy"""
            return self.getEnergy()

        @property
        def parts(self):
            """bonded particles"""
            return self.getParts()

        @property
        def potential(self):
            """bond potential"""
            return self.getPotential()

        @property
        def id(self):
            """bond id"""
            return self.getId()

        @property
        def dissociation_energy(self):
            """bond dissociation energy"""
            return self.getDissociationEnergy()

        @dissociation_energy.setter
        def dissociation_energy(self, dissociation_energy):
            self.setDissociationEnergy(dissociation_energy)

        @property
        def half_life(self):
            return self.getHalfLife()

        @half_life.setter
        def half_life(self, half_life):
            return self.setHalfLife(half_life)

        @property
        def active(self):
            """active flag"""
            return self.getActive()

        @property
        def style(self):
            """bond style"""
            return self.getStyle()
    %}
}

%pythoncode %{
    Bond = MxBond
    BondHandle = MxBondHandle
%}
