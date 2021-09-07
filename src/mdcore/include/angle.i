%{
    #include "angle.h"

%}

%rename(_getitem) MxAngleHandle::operator[];

%include "angle.h"

%template(vectorAngleHandle_p) std::vector<MxAngleHandle*>;

%extend MxAngleHandle {
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
        def active(self):
            """active flag"""
            return self.getActive()
    %}
}

%pythoncode %{
    Angle = MxAngle
    AngleHandle = MxAngleHandle
%}
