%{
    #include "rendering/MxClipPlane.hpp"

%}

%include "MxClipPlane.hpp"

%extend MxClipPlane {
    %pythoncode %{
        @property
        def point(self) -> MxVector3f:
            return self.getPoint();

        @property
        def normal(self) -> MxVector3f:
            return self.getNormal()

        @property
        def equation(self) -> MxVector4f:
            return self.getEquation()

        @equation.setter
        def equation(self, _equation):
            if isinstance(_equation, list):
                p = MxVector4f(_equation)
            elif isinstance(_equation, MxVector4f):
                p = _equation
            else:
                p = MxVector4f(list(_equation))
            return self.setEquation(p)
    %}
}

%extend MxClipPlanes {
    %pythoncode %{
        def __len__(self) -> int:
            return self.len()

        def __getitem__(self, item: int):
            return self.getClipPlaneEquation(item)

        def __setitem__(self, item: int, val):
            self.setClipPlaneEquation(item, val)
    %}
}


%pythoncode %{
    ClipPlane = MxClipPlane
    ClipPlanes = MxClipPlanes
    plane_equation = MxPlaneEquation
    parse_plane_equation = MxParsePlaneEquation
%}
