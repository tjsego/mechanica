.. _forces:

.. py:currentmodule:: mechanica

Forces
-------

Forces cause objects to move. In addition to forces that result from various
processes (*e.g.*, interactions via a :ref:`potential <potentials>`),
Mechanica also supports modeling explicit forces using a suite of
built-in forces, as well as custom forces. An instance of any of the built-in
forces can be created using a static method on the :py:attr:`Force` class
(:class:`MxForce` in C++), which can be :ref:`bound <binding>` to particles
by particle type.

.. _creating_forces-label:

Creating Forces
^^^^^^^^^^^^^^^^

:py:attr:`Force` objects are created simply by calling one of the static methods
on the :py:attr:`Force` class. For example, a random force can be created for
adding noise to the trajectory of particles, ::

    import mechanica as mx
    force = mx.Force.random(0.0, 1.0)

Custom forces can be created with the :py:attr:`ConstantForce` class
(:class:`MxConstantForce` in C++). A custom force requires a function
that takes no arguments and returns a three-component container of
floats that represent the current force whenever the function is called.
Mechanica will convert the function into a force that acts on whatever
particles are instructed in subsequent calls. For example, to create a
time-varying force in Python, ::

    import mechanica as mx
    import numpy as np
    ...
    force = mx.ConstantForce(lambda: [0.3, 1 * np.sin(0.4 * mx.Universe.time), 0], 0.01)

A :py:attr:`Force` instance can also be created by adding two existing
instances. Such operations can be arbitrarily performed to construct complicated
forces consisting of multiple constituent forces, ::

    force_noisy = mx.Force.random(0, 50)
    force_tstat = mx.Force.berendsen_tstat(10)
    force_noisy_tstat = force_noisy + force_tstat

.. note::

    Changes to constituent forces during simulation are reflected in forces
    that have been constructed from them using summation operations.

Built-in Forces
^^^^^^^^^^^^^^^^

Presently, the following built-in forces are supported, with corresponding
constructor method. For details on the parameters of each function, refer to the
:ref:`Mechanica API Reference <api_reference>`.

* Berendsen thermostat: :meth:`Force.berendsen_tstat <MxForce.berendsen_tstat>`
* Friction: :meth:`Force.friction <MxForce.friction>`
* Random: :meth:`Force.random <MxForce.random>`

Manipulating Forces on Particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aside from creating and applying forces on the basis of
:ref:`particle types <creating_particles_and_types>`, Mechanica also provides
fine-grained access to manipulating forces on individual particles.
The net force on each particle is accessible with the particle property
:attr:`force <MxParticleHandle.force>`, which returns the current net force
acting on the particle. While reading this property is valid, access to set the
value of :attr:`force <MxParticleHandle.force>` is not provided, as
Mechanica resets it at the beginning of each simulation step. However, Mechanica
provides read and write access to the vector to which the force on a particle
is reset at the beginning of each simulation step. The particle property
:attr:`force_init <MxParticleHandle.force_init>` contains the vector value of
the force on the particle *before* any processes acting on it are considered,
which is added to all force calculations during the simulation step. ::

    class MyParticleType(mx.ParticleType):
        pass

    ptype = MyParticleType.get()
    # Make lots of particles, but apply a force to only one of them
    parts = [ptype() for _ in range(100)]
    parts[0].force_init = [1, 2, 3]
