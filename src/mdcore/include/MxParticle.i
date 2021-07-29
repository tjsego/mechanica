%{
    #include "MxParticle.h"

%}

// Currently swig isn't playing nicely with MxVector3f pointers for MxParticleType::operator(), 
//  so we'll handle them manually for now
%rename(_call) MxParticleType::operator();
%rename(to_cluster) MxParticleHandle::operator MxClusterParticleHandle*();

%include "MxParticle.h"

%template(vectorParticle) std::vector<MxParticle>;
%template(vectorParticleHandle) std::vector<MxParticleHandle>;
%template(vectorParticleType) std::vector<MxParticleType>;

%extend MxParticleHandle {
    %pythoncode %{
        @property
        def charge(self):
            return self.getCharge()

        @charge.setter
        def charge(self, charge):
            self.setCharge(charge)

        @property
        def mass(self):
            return self.getMass()

        @mass.setter
        def mass(self, mass):
            self.setMass(mass)

        @property
        def frozen(self):
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def style(self):
            return self.getStyle()

        @style.setter
        def style(self, style):
            self.setStyle(style)

        @property
        def age(self):
            return self.getAge()

        @property
        def radius(self):
            return self.getRadius()

        @radius.setter
        def radius(self, radius):
            self.setRadius(radius)

        @property
        def name(self):
            return self.getName()

        @property
        def name2(self):
            return self.getName2()

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
        def id(self):
            return self.getId()

        @property
        def type_id(self):
            return self.getTypeId()

        @property
        def flags(self):
            return self.getFlags()

        @property
        def species(self):
            return self.getSpecies()

        @property
        def bonds(self):
            return self.getBonds()
    %}
}

%extend MxParticleType {
    %pythoncode %{
        def __call__(self, position=None, velocity=None, cluster_id=None):
            pos, vel = None, None
            if position is not None:
                pos = MxVector3f(list(position))

            if velocity is not None:
                vel = MxVector3f(list(velocity))

            return self._call(pos, vel, cluster_id)

        @property
        def frozen(self):
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def temperature(self):
            return self.getTemperature()

        @property
        def target_temperature(self):
            return self.getTargetTemperature()
    %}
}

%pythoncode %{
    Particle = MxParticle
    ParticleHandle = MxParticleHandle

    Newtonian = PARTICLE_NEWTONIAN
    Overdamped = PARTICLE_OVERDAMPED
%}

// In python, we'll specify particle types using class attributes of the same name 
//  as underlying C++ struct. ParticleType is the helper class through which this 
//  functionality occurs. ParticleType is defined in particle_type.py

