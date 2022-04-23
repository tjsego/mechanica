%{
    #include "dihedral.h"

%}

%rename(_getitem) MxDihedralHandle::operator[];

%ignore MxDihedral_StylePtr;

%include "dihedral.h"

%template(vectorDihedralHandle_p) std::vector<MxDihedralHandle*>;

%extend MxDihedral {
    %pythoncode %{
        def __reduce__(self):
            return MxDihedral.fromString, (self.toString(),)
    %}
}

%extend MxDihedralHandle {
    %pythoncode %{
        def __getitem__(self, index: int):
            return self._getitem(index)

        def __str__(self) -> str:
            return self.str()

        @property
        def energy(self):
            """angle energy"""
            return self.getEnergy()

        @property
        def parts(self):
            """bonded particles"""
            return self.getParts()

        @property
        def potential(self):
            """angle potential"""
            return self.getPotential()

        @property
        def id(self):
            """angle id"""
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
            """angle half life"""
            return self.getHalfLife()

        @half_life.setter
        def half_life(self, half_life):
            self.setHalfLife(half_life)

        @property
        def active(self):
            """active flag"""
            return self.getActive()

        @property
        def style(self):
            """angle style"""
            return self.getStyle()

        @style.setter
        def style(self, style):
            self.setStyle(style)

        @property
        def age(self):
            """angle age"""
            return self.getAge()
    %}
}

%pythoncode %{
    Dihedral = MxDihedral
    DihedralHandle = MxDihedralHandle
%}
