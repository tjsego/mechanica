%{
    #include "MxParticleList.hpp"

%}

%template(vectorParticleList_p) std::vector<MxParticleList*>;
%template(vector2ParticleList_p) std::vector<std::vector<MxParticleList*>>;
%template(vector3ParticleList_p) std::vector<std::vector<std::vector<MxParticleList*>>>;

%include "MxParticleList.hpp"

%extend MxParticleList {
    %pythoncode %{
        def __len__(self) -> int:
            return self.nr_parts

        def __getitem__(self, i: int):
            if i >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self.item(i)

        @property
        def virial(self):
            """Virial tensor of particles in list"""
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            """Radius of gyration of particles in list"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass of particles in list"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid of particles in list"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia of particles in list"""
            return self.getMomentOfInertia()

        @property
        def positions(self):
            """Position of each particle in list"""
            return self.getPositions()

        @property
        def velocities(self):
            """Velocity of each particle in list"""
            return self.getVelocities()

        @property
        def forces(self):
            """Net forces acting on each particle in list"""
            return self.getForces()

        def __reduce__(self):
            return MxParticleList.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    ParticleList = MxParticleList
%}
