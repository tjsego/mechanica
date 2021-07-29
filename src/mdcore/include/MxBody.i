%{
    #include "MxBody.hpp"

%}

%include "MxBody.hpp"

%extend MxBodyHandle {
    %pythoncode %{

        @property
        def position(self):
            return self.getPosition()

        @position.setter
        def position(self, position):
            self.setPosition(position)

        @property
        def velocity(self):
            return self.getVelocity()

        @velocity.setter
        def velocity(self, velocity):
            self.setVelocity(velocity)

        @property
        def force(self):
            return self.getForce()

        @force.setter
        def force(self, force):
            self.setForce(force)

        @property
        def orientation(self):
            return self.getOrientation()

        @orientation.setter
        def orientation(self, orientation):
            self.setOrientation(orientation)

        @property
        def spin(self):
            return self.getSpin()

        @spin.setter
        def spin(self, spin):
            self.setSpin(spin)

        @property
        def torque(self):
            return self.getTorque()

        @torque.setter
        def torque(self, torque):
            return self.setTorque(torque)

        @property
        def species(self):
            return self.getSpecies()

    %}
}

%pythoncode %{
    Body = MxBody
    BodyHandle = MxBodyHandle
%}
