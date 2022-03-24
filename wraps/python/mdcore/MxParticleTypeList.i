%{
    #include "MxParticleTypeList.h"

%}

%include "MxParticleTypeList.h"

%extend MxParticleTypeList {
    %pythoncode %{
        def __len__(self) -> int:
            return self.nr_parts

        def __getitem__(self, i: int):
            if i >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self.item(i)

        @property
        def virial(self):
            """Virial tensor of particles corresponding to all types in list"""
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            """Radius of gyration of particles corresponding to all types in list"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass of particles corresponding to all types in list"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid of particles corresponding to all types in list"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia of particles corresponding to all types in list"""
            return self.getMomentOfInertia()

        @property
        def positions(self):
            """Position of each particle corresponding to all types in list"""
            return self.getPositions()

        @property
        def velocities(self):
            """Velocity of each particle corresponding to all types in list"""
            return self.getVelocities()

        @property
        def forces(self):
            """Total net force acting on each particle corresponding to all types in list"""
            return self.getForces()

        def __reduce__(self):
            return MxParticleTypeList.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    ParticleTypeList = MxParticleTypeList
%}
