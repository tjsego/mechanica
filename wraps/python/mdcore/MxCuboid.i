%{
    #include "MxCuboid.hpp"

%}

%rename(_create) MxCuboid::create;

%include "MxCuboid.hpp"

%extend MxCuboid {
    %pythoncode %{
        @staticmethod
        def create(pos=None, size=None, orientation=None):
            if pos is not None and not isinstance(pos, MxVector3f):
                pos = MxVector3f(pos)
            if size is not None and not isinstance(size, MxVector3f):
                size = MxVector3f(size)
            if orientation is not None and not isinstance(orientation, MxVector3f):
                orientation = MxVector3f(orientation)
            return MxCuboid._create(pos, size, orientation)
    %}
}

%pythoncode %{
    Cuboid = MxCuboid
%}
