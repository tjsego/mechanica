.. _binding:

Binding
-------

Binding objects and processes together is one of the key ways to create a
Mechanica simulation. Binding connects a process (*e.g.*, a
:ref:`potential <potentials>`, :ref:`force <forces>`) with one
or more objects that the process acts on.
Binding in Mechanica is done with static methods on the class
:class:`Bind` (:class:`MxBind` in C++), and methods that implement
binding of processes to objects, in general, only return a code
indicating success or failure of the binding procedure.
For convenience, the :class:`Bind` class in Python is a top-level
variable named :attr:`bind`.

Binding Interactions Between Particles by Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An interaction between particles by pairs of types can be implemented
using the :class:`Bind` method :meth:`types`. ::

    import mechanica as mx
    ...
    # Bind an interaction between particles of type
    #   "A" and "B" according to the potential "pot"
    mx.bind.types(pot, A, B)

Binding Interactions Between Particles by Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An interaction between two particles can be implemented
using the :class:`Bind` method :meth:`particles`, which creates
a :ref:`bond <bonded_interactions>`. ::

    # Bind an interaction between particles "p0" and "p1"
    #   according to the potential "pot_bond"
    mx.bind.particles(pot_bond, p0, p1)

.. _binding_with_clusters:

Binding Interactions within and Between Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding an interaction between particles by pairs of types
can be particularized to only occuring between particles of
the same :ref:`cluster <clusters-label>`. The :class:`Bind` method
:meth:`types` provides a fourth, optional argument ``bound`` that,
when set to ``True``, only binds an interaction between particles
of a pair of types that are in the same cluster. ::

    # Bind an interaction between particle types "A" and "B" in the
    #   same cluster according to potential "potb"
    mx.bind.types(potb, A, B, bound=True)

.. _binding_boundaries_and_types:

Binding Interactions with Boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mechanica supports enforcing :ref:`boundary conditions <boundary>` on
particles as an interaction between a particle and a boundary according
to a potential. Binding an interaction between a particle by type and a
boundary can be implemented using the :class:`Bind` method
:meth:`boundaryCondition`. ::

    mx.init(bc={'top': 'potential'})
    ...
    # Bind an interaction between the top boundary and particle type
    #   "A" according to potential "pot"
    mx.bind.boundaryCondition(pot, mx.Universe.boundary_conditions.top, A)

Binding Forces to Particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding a :ref:`force <forces>` to a particle type can be implemented
using the :class:`Bind` method :meth:`force`. ::

    # Bind force "f" to act on particles of type "C"
    mx.bind.force(f, C)

