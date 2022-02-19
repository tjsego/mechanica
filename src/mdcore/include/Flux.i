%{
    #include "Flux.hpp"

%}

%ignore MxFluxes_integrate;
%ignore MxFluxes::fluxes;

%include "Flux.hpp"

%extend MxFluxes {
    %pythoncode %{
        def __reduce__(self):
            return MxFluxes.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    Flux = MxFlux
    Fluxes = MxFluxes
%}
