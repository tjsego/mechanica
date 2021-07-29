%{
    #include "bond.h"

%}

%rename(_getitem) MxBondHandle::operator[];

%include "bond.h"

%extend MxBondHandle {
    %pythoncode %{
        def __getitem__(self, index: int):
            return self._getitem(index)

        def __str__(self) -> str:
            return self.str()

        @property
        def energy(self):
            return self.getEnergy()

        @property
        def parts(self):
            return self.getParts()

        @property
        def potential(self):
            return self.getPotential()

        @property
        def id(self):
            return self.getId()

        @property
        def dissociation_energy(self):
            return self.getDissociationEnergy()

        @property
        def active(self):
            return self.getActive()

        @property
        def style(self):
            return self.getStyle()
    %}
}

// todo: define bond list from std::vector<MxBond*>

%pythoncode %{
    Bond = MxBond
%}
