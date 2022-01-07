%{
    #include "Flux.hpp"

%}

%ignore MxFluxes_integrate;

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
