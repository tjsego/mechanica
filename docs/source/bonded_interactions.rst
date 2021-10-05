.. _bonded_interactions:

Bonded Interactions
--------------------

A bonded interaction is an interaction due to a *bond* between
a group of particles. A bond describes an interaction exclusively
between the group of particles using a :ref:`potential <potentials>`.
Mechanica currently supports :ref:`bond <bonded_interactions:Bonds>`
and bond-like :ref:`angle <bonded_interactions:Angles>` and
:ref:`dihedral <bonded_interactions:Dihedrals>`.

Bonds
^^^^^^

.. image:: bond.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 78px

A bond describes an interaction between two particles in terms
of the distance between the two particles. A bond can be
created using the method :meth:`create` on the class
:class:`Bond` (:class:`MxBond` in C++), which returns a handle
to the newly created bond. A bond can be manually destroyed
using the :class:`Bond` method :meth:`destroy`.

.. rst-class::  clear-both

.. code-block:: python

    import mechanica as mx
    # Create a bond between particles "p0" and "p1" using the potential "pot_bond"
    bond_handle = mx.Bond.create(pot_bond, p0, p1)

:class:`Bond` instances have an optional dissociation energy
that, when set, describes an energy threshold above which the
bond is automatically destroyed. Likewise, each :class:`Bond`
instance has an optional half life that, when set, describes
the probability of destroying the bond at each simulation step,
which Mechanica automatically implements,

.. code-block:: python

    bond_handle.dissociation_energy = 1E-3
    bond_handle.half_life = 10.0

All bonds in the universe are accessible using the :class:`Universe`
method :meth:`Universe.bonds`,

.. code-block:: python

    all_bonds = mx.Universe.bonds()  # Get updated list of all bonds

Angles
^^^^^^^

.. image:: angle.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 125px

An angle describes an interaction between two particles in terms
of the angle made by their relative position vectors with respect
to a third particle. An angle can be created using the method
:meth:`create` on the class :class:`Angle` (:class:`MxAngle` in
C++), which returns a handle to the newly created angle.
An angle can be manually destroyed using the :class:`Angle`
method :meth:`destroy`. :class:`Angle` instances have analogous
properties and methods to most of those defined for :class:`Bond`
instances, including accessing each constituent particle
by indexing, and optional dissociation energy and half life.
All angles in the universe are accessible using the :class:`Universe`
method :meth:`Universe.angles`,

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between particles "p0" and "p2" w.r.t.
    #   particle "p1" using the potential "pot_ang"
    angle_handle = mx.Bond.create(pot_ang, p0, p1, p2)
    all_angles = mx.Universe.angles()  # Get updated list of all angles

Dihedrals
^^^^^^^^^^

.. image:: dihedral.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 157px

A dihedral describes an interaction between four particles in terms
of the angle between the planes made by their relative position vectors.
A dihedral can be created using the method :meth:`create` on the class
:class:`Dihedral` (:class:`MxDihedral` in C++), which returns a handle
to the newly created dihedral. A dihedral can be manually destroyed using
the :class:`Dihedral` method :meth:`destroy`. :class:`Dihedral` instances
have analogous properties and methods to most of those defined for
:class:`Bond` instances, including accessing each constituent particle
by indexing, and optional dissociation energy and half life.
All dihedrals in the universe are accessible using the :class:`Universe`
method :meth:`Universe.dihedrals`,

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between the plane made by particles "p0", "p1" and "p2"
    #   and the plane made by particles "p1", "p2" and "p3"
    #   using the potential "pot_dih"
    dihedral_handle = mx.Dihedral.create(pot_dih, p0, p1, p2, p3)
    all_dihedrals = mx.Universe.dihedrals()  # Get updated list of all dihedrals
