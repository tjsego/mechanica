%{
    #include "MxParticle.h"

%}

// Currently swig isn't playing nicely with MxVector3f pointers for MxParticleType::operator(), 
//  so we'll handle them manually for now
%rename(_call) MxParticleType::operator();
%rename(_factory) MxParticleType::factory(unsigned int, std::vector<MxVector3f>*, std::vector<MxVector3f>*, std::vector<int>*);
%rename(to_cluster) MxParticleHandle::operator MxClusterParticleHandle*();
%rename(to_cluster) MxParticleType::operator MxClusterParticleType*();

%ignore MxParticle_Colors;

%include "MxParticle.h"

%template(vectorParticle) std::vector<MxParticle>;
%template(vectorParticleHandle) std::vector<MxParticleHandle>;
%template(vectorParticleType) std::vector<MxParticleType>;

%extend MxParticle {
    %pythoncode %{
        def __reduce__(self):
            return MxParticle.fromString, (self.toString(),)
    %}
}

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
            if style.thisown:
                style.thisown = False
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

        @property
        def force_init(self):
            """Persistent force acting on particle"""
            return self.getForceInit()

        @force_init.setter
        def force_init(self, force):
            self.setForceInit(MxVector3f(force))

        @property
        def id(self):
            """Particle id"""
            return self.getId()

        @property
        def type_id(self):
            """Particle type id"""
            return self.getTypeId()

        @property
        def cluster_id(self):
            """Cluster particle id, if any; -1 if particle is not in a cluster"""
            return self.getClusterId()

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
        def __call__(self, *args, **kwargs):
            position = kwargs.get('position')
            velocity = kwargs.get('velocity')
            part_str = kwargs.get('part_str')
            cluster_id = kwargs.get('cluster_id')
            
            n_args = len(args)
            if n_args > 0:
                if isinstance(args[0], str):
                    part_str = args[0]
                    if n_args > 1:
                        if isinstance(args[1], int):
                            cluster_id = args[1]
                        else:
                            raise TypeError
                elif isinstance(args[0], int):
                    cluster_id = args[0]
                else:
                    position = args[0]
                    if n_args > 1:
                        if isinstance(args[1], int):
                            cluster_id = args[1]
                        else:
                            velocity = args[1]
                            if n_args > 2:
                                cluster_id = args[2]

            pos, vel = None, None
            if position is not None:
                pos = MxVector3f(list(position)) if not isinstance(position, MxVector3f) else position

            if velocity is not None:
                vel = MxVector3f(list(velocity)) if not isinstance(velocity, MxVector3f) else velocity

            if part_str is not None:
                return self._call(part_str, cluster_id)
            return self._call(pos, vel, cluster_id)

        def factory(self, nr_parts=0, positions=None, velocities=None, cluster_ids=None):
            _positions = None
            if positions is not None:
                _positions = vectorMxVector3f()
                [_positions.push_back(MxVector3f(x)) for x in positions]

            _velocities = None
            if velocities is not None:
                _velocities = vectorMxVector3f()
                [_velocities.push_back(MxVector3f(x)) for x in velocities]

            _cluster_ids = None
            if cluster_ids is not None:
                _cluster_ids = vectori()
                [_cluster_ids.push_back(x) for x in cluster_ids]

            return self._factory(nr_parts=nr_parts, positions=_positions, velocities=_velocities, clusterIds=_cluster_ids)

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

        def __reduce__(self):
            return MxParticleType.fromString, (self.toString(),)
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

