%{
    #include "rendering/MxClipPlane.hpp"

%}

%include "MxClipPlane.hpp"

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
    ClipPlanes = MxClipPlanes
    plane_equation = MxPlaneEquation
    parse_plane_equation = MxParsePlaneEquation
%}
