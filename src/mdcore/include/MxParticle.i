%{
    #include "MxParticle.h"

%}

// Currently swig isn't playing nicely with MxVector3f pointers for MxParticleType::operator(), 
//  so we'll handle them manually for now
%rename(_call) MxParticleType::operator();
%rename(to_cluster) MxParticleHandle::operator MxClusterParticleHandle*();
%rename(to_cluster) MxParticleType::operator MxClusterParticleType*();

%include "MxParticle.h"

%template(vectorParticle) std::vector<MxParticle>;
%template(vectorParticleHandle) std::vector<MxParticleHandle>;
%template(vectorParticleType) std::vector<MxParticleType>;

%extend MxParticleHandle {
    %pythoncode %{
        @property
        def charge(self):
            """Particle charge"""
            return self.getCharge()

        @charge.setter
        def charge(self, charge):
            self.setCharge(charge)

        @property
        def mass(self):
            """Particle mass"""
            return self.getMass()

        @mass.setter
        def mass(self, mass):
            self.setMass(mass)

        @property
        def frozen(self):
            """Particle frozen flag"""
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            """Particle frozen flag along x"""
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            """Particle frozen flag along y"""
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            """Particle frozen flag along z"""
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def style(self):
            """Particle style"""
            return self.getStyle()

        @style.setter
        def style(self, style):
            self.setStyle(style)

        @property
        def age(self):
            """Particle age"""
            return self.getAge()

        @property
        def radius(self):
            """Particle radius"""
            return self.getRadius()

        @radius.setter
        def radius(self, radius):
            self.setRadius(radius)

        @property
        def name(self):
            """Particle name"""
            return self.getName()

        @property
        def name2(self):
            return self.getName2()

        @property
        def position(self):
            """Particle position"""
            return self.getPosition()

        @position.setter
        def position(self, position):
            self.setPosition(MxVector3f(position))

        @property
        def velocity(self):
            """Particle velocity"""
            return self.getVelocity()

        @velocity.setter
        def velocity(self, velocity):
            self.setVelocity(MxVector3f(velocity))

        @property
        def force(self):
            """Net force acting on particle"""
            return self.getForce()

        @force.setter
        def force(self, force):
            self.setForce(MxVector3f(force))

        @property
        def id(self):
            """Particle id"""
            return self.getId()

        @property
        def type_id(self):
            """Particle type id"""
            return self.getTypeId()

        @property
        def flags(self):
            return self.getFlags()

        @property
        def species(self):
            """Particle species"""
            return self.getSpecies()

        @property
        def bonds(self):
            """Bonds attached to particle"""
            return self.getBonds()

        @property
        def angles(self):
            """Angles attached to particle"""
            return self.getAngles()

        @property
        def dihedrals(self):
            """Dihedrals attached to particle"""
            return self.getDihedrals()
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
            """Particle type frozen flag"""
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            """Particle type frozen flag along x"""
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            """Particle type frozen flag along y"""
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            """Particle type frozen flag along z"""
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def temperature(self):
            """Particle type temperature"""
            return self.getTemperature()

        @property
        def target_temperature(self):
            """Particle type target temperature"""
            return self.getTargetTemperature()

        @target_temperature.setter
        def target_temperature(self, temperature):
            self.setTargetTemperature(temperature)
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

