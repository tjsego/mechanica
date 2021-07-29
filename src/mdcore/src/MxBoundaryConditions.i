%{
    #include "MxBoundaryConditions.hpp"

%}

%include "MxBoundaryConditions.hpp"

%extend MxBoundaryCondition {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        @property
        def kind_str(self) -> str:
            return self.kindStr()
    %}
}


%extend MxBoundaryConditions {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()
    %}
}

%pythoncode %{
    BoundaryCondition = MxBoundaryCondition
    BoundaryConditions = MxBoundaryConditions

    BOUNDARY_NONE = space_periodic_none
    PERIODIC_X = space_periodic_x
    PERIODIC_Y = space_periodic_y
    PERIODIC_Z = space_periodic_z
    PERIODIC_FULL = space_periodic_full
    PERIODIC_GHOST_X = space_periodic_ghost_x
    PERIODIC_GHOST_Y = space_periodic_ghost_y
    PERIODIC_GHOST_Z = space_periodic_ghost_z
    PERIODIC_GHOST_FULL = space_periodic_ghost_full
    FREESLIP_X = SPACE_FREESLIP_X
    FREESLIP_Y = SPACE_FREESLIP_Y
    FREESLIP_Z = SPACE_FREESLIP_Z
    FREESLIP_FULL = SPACE_FREESLIP_FULL
%}
// todo: verify enums are properly named