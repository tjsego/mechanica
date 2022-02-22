.. _creating_particles_and_types:

.. py:currentmodule:: mechanica

Creating Particles and Particle Types
======================================

The *particle* is the most basic physical object of Mechanica.
A particle occupies a point in space and necessarily has a mass.
Forces acts on a particle and cause a corresponding change in its
trajectory according to the dynamics of a simulation.
A particle has a specified radius for various modeling and
visualization purposes, but has no intrinsic volume.
Like the radius property, a particle has other properties
that can be assigned to it that have no meaning except within
the context of a model.
Depending on the application, a particle can represent an atom,
a molecule, a point on a surface or in a lattice, a part of a cell,
a cell, a parcel of material, or something else with a dynamic position.
Particles in Mechanica can be created and destroyed, interact
with each other and other objects, and even carry cargo that
can be transported between particles.

Particle Types
---------------

The only meaning that Mechanica assigns to particles is that each
particle is an instance of a *particle type*. A particle type defines
a template definition from which particles can be created, and a
particle type is an identifier of a corresponding group of particles
in a simulation such that all particles can always be grouped according
to a set of particle types. A particle can :ref:`change type <become>`,
and particle processes, interactions and properties in Mechanica can be
specified on the basis of individual particles, but each particle
always has a type, and the particle type defines the initial
properties of the particle that the particle type can create.

As such, the basic specification of particle properties in Mechanica
occurs through the specification of particle types.
Every particle type is a subclass of the Mechanica class
:class:`MxParticleType`. Each particle type in a simulation is an
object in memory that is instantiated and can be used for various tasks
(*e.g.*, to create particles). However, unlike a particle,
an instance of a particle type is not a model object, but rather
a dynamic model definition and simulation entry point for specifying
particle properties on the basis of type (*e.g.*,
:ref:`particle dynamics <making_things_move>`)
and doing type-related operations (*e.g.*,
:ref:`finding all particles of a type <accessing>`).
Furthermore, Mechanica only recognizes particle types that have
been registered with Mechanica for a simulation, and there is exactly
one instance of each registered particle type that Mechanica works
with during a simulation.
:class:`MxParticleType` and each subclass of it has a special method
:py:meth:`get <MxParticleType.get>` that retrieves the instance of the
particle type that Mechanica is using in a simulation (assuming the type
has been registered), which is not necessarily an instance of a particle
type that might be created during simulation.
This way, the working instance of a registered particle type can be
retrieved from Mechanica and interacted with by referring to the
corresponding class definition of the particle type.

Mechanica provides a few ways to create and register particle types.
The recommend method of creating a particle type is to create a class
definition that is a subclass of :class:`MxParticleType`. The properties
of a particle type can be changed at anytime during a simulation without
issue, with the exception of its name. For many reasons related to
computational performance and model accessibility, the name of a
particle type can be set during a simulation. However, doing so after
registering a particle type will result in undefined behavior and
likely cause a simulation to fail.

.. note:

    Changing the properties of a type only affects particles created
    thereafter using the particle type. Changes to the properties of
    existing particles must be done using operations on the particles.

In Python, particle types can be specified using a class definition
that derives from the class :class:`ParticleType`, and properties
can be specified as class attributes,

.. code-block:: python

    import mechanica as mx

    class MyParticleType(mx.ParticleType):
        mass = 1.0
        radius = 1.0
        dynamics = mx.Overdamped

A new particle type can be simultaneously instantiated and registered
with Mechanica in Python with the class method :meth:`ParticleType.get`,

.. code-block:: python

    my_particle_type = MyParticleType.get()

.. note::

    In Python, :class:`ParticleType` is not the same as
    :class:`MxParticleType`. Rather, it is a convenience class that
    automates the process of creating, registering and retrieving a
    :class:`MxParticleType` instance with Mechanica using the class method
    :meth:`get <ParticleType.get>`, which always returns the actual registered
    :class:`MxParticleType` instance without ambiguity. A
    :class:`ParticleType` instance can be instantiated in the typical
    way and operated on without any need for the Mechanica engine, so
    long as :meth:`get <ParticleType.get>` is not called on the instance.
    Furthermore, additional specifications can be made on a :class:`ParticleType`
    class definition. However, ``self`` in a :class:`ParticleType` method
    does not refer to the corresponding :class:`MxParticleType` instance
    registered with Mechanica.

Particle type definitions can then be changed on-the-fly in Python for
particles created later in simulation,

.. code-block:: python

    # Changing back to default dynamics!
    my_particle_type.dynamics = mx.Newtonian

In C++, particle types can be specified using a class definition
that derives from the class :class:`MxParticleType`, and properties
can be specified as members during instantiation,

.. code-block:: cpp

    #include <MxParticleType.h>

    struct MyParticleType : MxParticleType {
        MyParticleType() : MxParticleType(true) {
            mass = 1.0;
            radius = 1.0;
            dynamics = PARTICLE_OVERDAMPED;
            registerType();
        }
    };

Note that :meth:`registerType` is how particle types are
registered with Mechanica. The call to :meth:`registerType` in the
constructor is optional, and can instead be called after
instantiation of the particle type (*i.e.*, subsequent attempts to
register the type are ignored).

A registered particle type can be retrieved from Mechanica in C++
with the class method :meth:`MxParticleType::get`, the returned
pointer of which is of type :class:`MxParticleType` that can be
safely cast to the new particle type (assuming no conflicting
additional specifications on the class definition),

.. code-block:: cpp

    MyParticleType *myParticleType = new MyParticleType();
    myParticleType = (MyParticleType*)myParticleType->get();

Particle type definitions can then be changed on-the-fly in C++ for
particles created later in simulation,

.. code-block:: cpp

    // Changing back to default dynamics!
    myParticleType->dynamics = PARTICLE_NEWTONIAN

A particle type can also be created on the fly using a unique name,
and the unique name can be used to retrieve the registered particle
type instance from Mechanica,

.. code-block:: python

    another_particle_type = mx.MxParticleType.newType('AnotherParticleType')
    another_particle_type.registerType()
    another_particle_type = mx.MxParticleType_FindFromName('AnotherParticleType')

Particles
----------

Particle type instances function like factories of particles.
Each particle type instance can be called like a function to
create exactly one new particle. Such a call returns a handle
to the newly created particle. Referring to the previous examples
in Python,

.. code-block:: python

    particle_handle = my_particle_type(position=[1.0, 2.0, 3.0], velocity=[0.0, 0.0, 1.0])
    # Change the mass of this particle
    particle_handle.mass = 0.5

When an initial position or velocity is not specified while creating
a particle, it is randomly selected. Initial position, when randomly
selected, is always within the universe. Initial velocity, when
randomly selected, has a random direction but speed such that the
initial kinetic energy of the particle is equal to the particle type
property ``target_temperature`` (in C++, ``target_energy``).

.. _clusters-label:

Clusters
---------

A *cluster* is a special kind of particle that contains other particles,
including other clusters, with a corresponding base *cluster type*.
The cluster and cluster type are extensions of the particle and
particle types, respectively, and so the same properties and methods
are available to each along with the additional descriptions supporting
this idea of a cluster.

In general, operations concerning particles and particle types are
not concerned with the distinction between a particle and a cluster.
As such, operations can return a cluster or cluster type instance
as a particle or particle type instance. To handle any ambiguity
about whether a particle is actually a cluster, each particle type
has a method :meth:`isCluster <MxParticleType.isCluster>` that
returns ``true`` if the particle type is a cluster type. In such cases,
corresponding particles can be safely cast to a cluster type in appropriate
languages. In Python, this cast can be accomplished with the particle method
:meth:`to_cluster <MxParticleType.to_cluster>`, which returns a cluster instance
of the same underlying particle. Likewise the :meth:`get <ClusterType.get>`
method of cluster types in Python correctly returns instances of the cluster type.

Clusters also have unique properties (*e.g.*, center of mass,
:ref:`particle inventory <accessing>`)
that are derived from their constituent particles, and unique processes
that involve their constituent particles (*e.g.*, :ref:`cleavage <cleavage>`).
Clusters can also interact with other particles on the basis of
particle types *and* :ref:`cluster ownership <binding_with_clusters>`.

Defining Clusters
^^^^^^^^^^^^^^^^^^

The class :class:`ClusterType` (:class:`MxClusterParticleType` in C++) corresponds
to :class:`ParticleType` for clusters, and the class :py:attr:`Cluster`
corresponds to :py:attr:`Particle`. New cluster types can be specified in
the same way as particle types, with the additional specification of
other particle and cluster types that constitute it with the
cluster type property :py:attr:`types <ClusterType.types>`.

.. code-block:: python

    import mechanica as mx

    class ConstitutiveType(mx.ParticleType):
        radius = 0.1

    constitutive_type = ConstitutiveType.get()

    class MyClusterType(mx.ClusterType):
        radius = 1.0
        types = [constitutive_type]

Creating with Clusters
^^^^^^^^^^^^^^^^^^^^^^^

Clusters have the unique ability to function like particle types in that
they, like particle types, can create new constituent particles, and in
a similar way. A particle of a constituent particle type of a cluster
can be created using the cluster in the same way as when creating with
a particle type, but while passing the constituent particle type to be
created. Constituent particles created by a particular cluster
belong to that cluster,

.. code-block:: python

    my_cluster_type = MyClusterType.get()
    cluster_particle = my_cluster_type(position=[1.0, 2.0, 3.0])
    const_part = cluster_particle(particle_type=my_cluster_type, position=[2.0, 2.0, 3.0])
