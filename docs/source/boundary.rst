.. _boundary:

Boundary Conditions
--------------------

.. py:currentmodule:: mechanica

Mechanica supports a number of boundary conditions on bases as detailed as individual
boundaries and particle types, to as generic as default conditions on all boundaries.
In Python, different boundary conditions can be specified via the argument ``bc`` to the
top-level :func:`init` method, where heterogeneous conditions are specified by
passing a dictionary, and homogeneous conditions are specified by passing a constant.
In C++, different boundary conditions can be specified by creating and configuring a
:class:`MxBoundaryConditionsArgsContainer` instance (defined in *MxBoundaryConditions.hpp*),
and then setting it on the member :attr:`universeConfig` of a :class:`MxSimulator_Config`
instance using the method :meth:`setBoundaryConditions` before
:ref:`initializing Mechanica <running_a_sim>` with :meth:`MxSimulator_initC`
(defined in *MxSimulator.h*).

In general, each boundary can be referred to with the names ``"left"`` and ``"right"``
for the lower and upper boundaries along the first spatial dimension,
``"bottom"`` and ``"top"`` for the lower and upper boundaries along the
second dimension, and ``"back"`` and ``"front"`` for the lower and upper boundaries
along the third dimension. Both boundaries along the first spatial dimension
can be reffered to with the name ``"x"``, along the second dimension as ``"y"``, and
along the third dimension as ``"z"``. Each type of boundary condition also has a
designated name, which can be referred to using a string, as well as a constant.

Periodic
^^^^^^^^^

The *periodic boundary condition* effectively simulates an infinite domain where any
agents that leave one side automatically appears at the opposite boundary. Also, agents
near a boundary can interact with the agents near the opposite boundary, (*e.g.*
a repulsive interaction can occur between agents near the left and right boundaries).
Periodic boundary conditions also determine how chemical
:ref:`fluxes operate <flux-label>`.
The periodic boundary condition can be employed using the name ``"periodic"`` and
constant ``BOUNDARY_PERIODIC``. ::

    import mechanica as mx
    mx.init(bc={'x':'periodic', 'z' : mx.BOUNDARY_PERIODIC})

Free-slip and No-slip
^^^^^^^^^^^^^^^^^^^^^^

*Free-slip* and *no-slip boundary conditions* reflect particles that impact a boundary
back into the simulation domain. Free-slip boundaries are essentially equivalent to a
boundary moving at the same tangential velocity *with* the simulation
objects, and can be thought of as each impacting agent colliding with an equivlent
ghost agent with the same tangent velocity at the boundary. No-slip boundaries are
equivalent to a stationary wall, in that impacting particles bounce straight back,
inverting their velocity.
Free-slip and no-slip boundary conditions can be employed using the names
``"freeslip"`` and ``"noslip"`` and constants ``BOUNDARY_FREESLIP`` and
``BOUNDARY_NO_SLIP``, respectively. ::

    mx.init(bc={'front':'freeslip', 'back' : mx.BOUNDARY_FREESLIP,
                'left': 'noslip', 'right': mx.BOUNDARY_NO_SLIP})


.. image:: boundary-conditions.png
    :alt: usage
    :width: 500px
    :class: sphx-glr-single-img

Velocity
^^^^^^^^^

A *velocity boundary condition* models a simulation domain with a moving boundary.
For example, the no-slip boundary condition is a particularization of the velocity boundary
conditon to zero velocity.
The velocity boundary condition can be employed with the name ``"velocity"``. ::

    m.init(bc={'top': {'velocity': [-1, 0, 0]})
  

Potential
^^^^^^^^^^

Mechanica supports implementing a *potential boundary condition* as an interaction between
a boundary and a particle type according to a :ref:`potential <potentials>`.
When a boundary condition is designated as a potential,
a potential can later be :ref:`bound <binding_boundaries_and_types>`
to the boundary and types of particles.
The potential boundary condition can be employed with the name ``"potential"``
and constant ``BOUNDARY_POTENTIAL``. ::

    mx.init(bc={'top': 'potential', 'bottom': mx.BOUNDARY_POTENTIAL})
    # Bind potentials later for the top and bottom boundaries!
