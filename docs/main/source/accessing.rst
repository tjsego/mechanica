.. _accessing:

.. py:currentmodule:: mechanica

Finding and Accessing Objects
------------------------------

One of the more common activities is finding, accessing and interacting
with model, simulation and system objects after their creation.
Mechanica provides a number of ways to retrieve various objects, from
particles that are created dynamically, to objects like the universe
that are created automatically when initializing Mechanica.

Finding Particles
^^^^^^^^^^^^^^^^^^

Mechanica methods that return a list of particles often return a
specialized list called a :py:attr:`ParticleList` (:class:`MxParticleList`
in C++). :py:attr:`ParticleList` is a special list that contains particles
and has a number of convenience methods for dealing with spatial
information (*e.g.*, calculating the center of mass of this list).
In cases where a function returns a more basic container
(*e.g.*, a Python list) of particles (or requires a :py:attr:`ParticleList`
as argument), a :py:attr:`ParticleList` can be easily constructed from most
basic containers by simply passing the container to the
:py:attr:`ParticleList` constructor.

Each :py:attr:`ParticleType`-derived particle type instance has a method
:py:meth:`ParticleType.items <docs_api_py:MxParticleType.items>` that returns a
list of all of the particles of that type::

    class MyType(ParticleType):
      ...

    # make ten particles...
    my_type = MyType.get()
    [my_type() for _ in range(10)]

    # get all the instances of that type:
    parts = my_type.items()

*All* particles currently in a simulation are also accessible from the
:ref:`universe <mechanica_universe>` via the method
:py:meth:`particles <MxUniverse.particles>`.
Likewise, all particles constituting a
:ref:`cluster <clusters-label>` can be accessed with the cluster method
:py:meth:`items <MxParticleType.items>`.

Each particle is aware if its neighbors, and we can get a list of
all the neghbors of an object by calling the particle method
:py:meth:`neighbors <MxParticleHandle.neighbors>`.
The :py:meth:`neighbors <MxParticleHandle.neighbors>` method accepts two
optional arguments, ``distance`` and ``types``. Distance is the distance away
from the *surface* of the present particle to search for neighbors, and types
is a list of particle types to restrict the search to. ::

    # Make a particle and find its nearby A- and B-type neighbors
    p = my_type()
    nbrs = p.neighbors(distance=1, types=[A, B])

Mechanica also provides the ability to organize all the particles into a
discretized grid, for example, to display some quantity as a function of
spatial position. The :ref:`universe <mechanica_universe>` method
:py:meth:`grid <MxUniverse.grid>` returns a three-dimensional container of all
particle lists with dimensions according to the shape passed to
:py:meth:`grid <MxUniverse.grid>`. The ordering of the passed shape is the same
as for position in space. The list at each index of the returned container
corresponds to the particles in each subspace of the discretized space according
to a regular lattice. For example, when discretizing space into a 8x9x10 grid,
the particles in the first subspace along the first dimension, second subspace
along the second dimension, and third subspace along the third dimension is
readily accessible, ::

    parts = Universe.grid([8, 9, 10])
    parts_ss = parts[0][1][2]
    print('Subspace velocities:', parts_ss.velocities)

Finding Bonds
^^^^^^^^^^^^^^

Like particles, :ref:`bonds and bond-like objects <bonded_interactions>`
can be dynamically created and destroyed, and Mechanica provides a number
of ways to retrieve them. All bonds and bond-like objects attached to a
particle can be retrieved using the property ``bonds`` and comparable, ::

    # Get all bond and bond-like objects attached to particle "p"
    bonds = p.bonds
    angles = p.bonds_angle

Likewise all bond and bond-like objects currently in the simulation

*All* bonds and bond-like objects currently in a simulation are also
accessible from the :ref:`universe <mechanica_universe>` using the method
:py:meth:`bonds <MxUniverse.bonds>` and comparable, ::

    # Get all bond and bond-like objects in the universe
    all_bonds = Universe.bonds()
    all_angles = Universe.bonds_angle()

Finding Simulation Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For convenience, both the :ref:`simulator and universe <running_a_sim>` are
available in Python as module-level variables, ::

    from mechanica import mx
    ...
    sim = mx.Simulator
    universe = mx.Universe

In C++, the simulator and universe can both be easily accessed,

.. code-block:: cpp

    #include <MxSimulator.h>
    #include <MxUniverse.h>
    ...
    MxSimulator *sim = MxSimulator::get();
    MxUniverse *universe = getUniverse();
