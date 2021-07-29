%{
    #include "MxUniverse.h"

%}

%include "MxUniverse.h"

%extend MxUniverse {
    %pythoncode %{
        @property
        def temperature(self):
            return self.getTemperature()

        @property
        def time(self):
            return self.getTime()

        @property
        def dt(self):
            return self.getDt()

        @property
        def event_list(self):
            return self.getEventList()

        @property
        def boundary_conditions(self):
            return self.getBoundaryConditions()

        @property
        def kinetic_energy(self):
            return self.getKineticEnergy()

        @property
        def center(self) -> MxVector3f:
            return self.getCenter()

        # Supporting old interface for now
        @staticmethod
        def bind(*args, **kwargs):
            print('Using old interface: recommend using interface defined on mechanica.bind')
            if len(args) == 4:
                return MxBind.types(*args)
            
            if len(args) == 3 and 'bound' in kwargs.keys():
                return MxBind.types(args[0], args[1], args[2], kwargs['bound'])

            if len(args) >= 2 and isinstance(args[0], MxForce):
                return MxBind.force(*args)

            if len(args) == 3 and isinstance(args[0], MxPotential):
                if isinstance(args[1], MxParticle) and isinstance(args[2], MxParticle):
                    return MxBind.particles(*args)
                if isinstance(args[1], MxParticleType) and isinstance(args[2], MxParticleType):
                    return MxBind.types(*args)

            raise ValueError("No valid combination of inputs")
    %}
}

%pythoncode %{
    Universe = getUniverse()

    reset_species = MxUniverse.resetSpecies
%}