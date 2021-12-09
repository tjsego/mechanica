%{
    #include "Flux.hpp"

%}

%ignore MxFluxes_integrate;

%include "Flux.hpp"

%pythoncode %{
    Flux = MxFlux
    Fluxes = MxFluxes
%}
