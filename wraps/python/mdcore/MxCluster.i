%{
    #include "MxCluster.hpp"

%}

// Currently swig isn't playing nicely with MxVector3f pointers for ::operator(), 
//  so we'll handle them manually for now
%rename(_call) MxClusterParticleHandle::operator();
%rename(_call) MxClusterParticleType::operator();

%include "MxCluster.hpp"

%template(list_ParticleType) std::list<MxParticleType*>;

%extend MxCluster {
    %pythoncode %{
        def __reduce__(self):
            return MxCluster_fromString, (self.toString(),)
    %}
}

%extend MxClusterParticleHandle {
    %pythoncode %{
        def __call__(self, particle_type, *args, **kwargs):
            position = kwargs.get('position')
            velocity = kwargs.get('velocity')
            part_str = kwargs.get('part_str')
            
            n_args = len(args)
            if n_args > 0:
                if isinstance(args[0], str):
                    part_str = args[0]
                    if n_args > 1:
                        raise TypeError
                else:
                    position = args[0]
                    if n_args > 1:
                        velocity = args[1]
                        if n_args > 2:
                            raise TypeError

            pos, vel = None, None
            if position is not None:
                pos = MxVector3f(list(position))

            if velocity is not None:
                vel = MxVector3f(list(velocity))

            args = []
            if pos is not None:
                args.append(pos)
            if vel is not None:
                args.append(vel)
            if args:
                return self._call(particle_type, *args)

            if part_str is not None:
                args.append(part_str)

            return self._call(particle_type, *args)

        @property
        def radius_of_gyration(self):
            """Radius of gyration"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia"""
            return self.getMomentOfInertia()
    %}
}

%extend MxClusterParticleType {
    %pythoncode %{
        def __call__(self, position=None, velocity=None, cluster_id=None):
            ph = MxParticleType.__call__(self, position, velocity, cluster_id)
            return MxClusterParticleHandle(ph.id, ph.type_id)

        def __reduce__(self):
            return MxClusterParticleType_fromString, (self.toString(),)
    %}
}

%pythoncode %{
    Cluster = MxCluster
    ClusterHandle = MxClusterParticleHandle
%}

// In python, we'll specify cluster types using class attributes of the same name 
//  as underlying C++ struct. ClusterType is the helper class through which this 
//  functionality occurs. ClusterType is defined in particle_type.py
