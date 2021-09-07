%{
    #include "MxBind.hpp"

%}

// Replacing these with more python-friendly methods
%ignore MxBind::bonds;
%ignore MxBind::sphere;

%include "MxBind.hpp"

%template(pairParticleList_BondHandle) std::pair<MxParticleList*, std::vector<MxBondHandle*>*>;
%template(pairParticleType_ParticleType) std::pair<MxParticleType*, MxParticleType*>;
%template(vectorPairParticleType_ParticleType) std::vector<std::pair<MxParticleType*, MxParticleType*>*>;

%extend MxBind {
    %pythoncode %{
        @staticmethod
        def bonds(potential, particles, cutoff, pairs=None, half_life=None, bond_energy=None, flags=0):
            if not isinstance(particles, MxParticleList):
                particle_list = MxParticleList()
                [particle_list.insert(p) for p in particles]
            else:
                particle_list = particles

            ppairs = None
            if pairs is not None:
                ppairs = vectorPairParticleType_ParticleType([pairParticleType_ParticleType(p) for p in pairs])
            return MxBind._bondsPy(potential, particle_list, cutoff, ppairs, half_life, bond_energy, flags)

        @staticmethod
        def sphere(potential, n, center=None, radius=None, phi=None, type=None):
            if phi is not None:
                phi0, phi1 = phi[0], phi[1]
            else:
                phi0, phi1 = None, None
            result = MxBind._spherePy(potential, n, center, radius, phi0, phi1, type)
            return result.first, result.second
    %}
}

%pythoncode %{
    bind = MxBind
%}
