%{
    #include "MxParticleList.hpp"

%}

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
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            return self.getCenterOfMass()

        @property
        def centroid(self):
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            return self.getMomentOfInertia()

        @property
        def positions(self):
            return self.getPositions()

        @property
        def velocities(self):
            return self.getVelocities()

        @property
        def forces(self):
            return self.getForces()
    %}
}

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
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            return self.getCenterOfMass()

        @property
        def centroid(self):
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            return self.getMomentOfInertia()

        @property
        def positions(self):
            return self.getPositions()

        @property
        def velocities(self):
            return self.getVelocities()

        @property
        def forces(self):
            return self.getForces()
    %}
}

%pythoncode %{
    ParticleList = MxParticleList
    ParticleTypeList = MxParticleTypeList
%}
