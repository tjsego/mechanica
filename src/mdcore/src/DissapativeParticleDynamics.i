%{
    #include "DissapativeParticleDynamics.hpp"

%}

%include "DissapativeParticleDynamics.hpp"

%extend DPDPotential {
    %pythoncode %{
        def __reduce__(self):
            return DPDPotential_fromStr, (self.toString(),)
    %}
}
