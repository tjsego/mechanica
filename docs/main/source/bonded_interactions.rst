.. _bonded_interactions:

.. py:currentmodule:: mechanica

Bonded Interactions
--------------------

.. figure:: nucleos_ta.png
    :width: 300px
    :alt: alternate text
    :align: center
    :figclass: align-center

    Mechanica models of thymine (left) and adenine (right) molecules
    using bonded interactions.

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
created using the method :py:meth:`create <MxBond.create>` on the class
:py:attr:`Bond` (:class:`MxBond` in C++), which returns a handle
to the newly created bond. A bond can be manually destroyed
using the :py:attr:`Bond` method :py:meth:`destroy <MxBondHandle.destroy>`.

.. rst-class::  clear-both

.. code-block:: python

    import mechanica as mx
    # Create a bond between particles "p0" and "p1" using the potential "pot_bond"
    bond_handle = mx.Bond.create(pot_bond, p0, p1)

:py:attr:`Bond` instances have an optional dissociation energy
that, when set, describes an energy threshold above which the
bond is automatically destroyed. Likewise, each :py:attr:`Bond`
instance has an optional half life that, when set, describes
the probability of destroying the bond at each simulation step,
which Mechanica automatically implements,

.. code-block:: python

    bond_handle.dissociation_energy = 1E-3
    bond_handle.half_life = 10.0

All bonds in the universe are accessible using the :py:attr:`Universe`
method :py:meth:`bonds <MxUniverse.bonds>`,

.. code-block:: python

    all_bonds = mx.Universe.bonds()  # Get updated list of all bonds

A bond is rendered as a line joining the two particles of the bond.

Angles
^^^^^^^

.. image:: angle.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 125px

An angle describes an interaction between two particles in terms
of the angle made by their relative position vectors with respect
to a third particle. An angle can be created using the method
:py:meth:`create <MxAngle.create>` on the class :py:attr:`Angle`
(:class:`MxAngle` in C++), which returns a handle to the newly
created angle. An angle can be manually destroyed using the :py:attr:`Angle`
method :py:meth:`destroy <MxAngleHandle.destroy>`. :py:attr:`Angle` instances
have analogous properties and methods to most of those defined for :py:attr:`Bond`
instances, including accessing each constituent particle
by indexing, and optional dissociation energy and half life.
All angles in the universe are accessible using the :py:attr:`Universe`
method :py:meth:`angles <MxUniverse.angles>`,

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between particles "p0" and "p2" w.r.t.
    #   particle "p1" using the potential "pot_ang"
    angle_handle = mx.Bond.create(pot_ang, p0, p1, p2)
    all_angles = mx.Universe.angles()  # Get updated list of all angles

An angle is rendered as a line joining the center particle and each end
particle, and a line joining the midpoint of those two lines.

Dihedrals
^^^^^^^^^^

.. image:: dihedral.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 157px

A dihedral describes an interaction between four particles in terms
of the angle between the planes made by their relative position vectors.
A dihedral can be created using the method :py:meth:`create <MxDihedral.create>`
on the class :py:attr:`Dihedral` (:class:`MxDihedral` in C++), which returns a handle
to the newly created dihedral. A dihedral can be manually destroyed using
the :py:attr:`Dihedral` method :py:meth:`destroy <MxDihedralHandle.destroy>`.
:py:attr:`Dihedral` instances have analogous properties and methods to most
of those defined for :py:attr:`Bond` instances, including accessing each constituent particle
by indexing, and optional dissociation energy and half life.
All dihedrals in the universe are accessible using the :py:attr:`Universe`
method :py:meth:`dihedrals <MxUniverse.dihedrals>`,

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between the plane made by particles "p0", "p1" and "p2"
    #   and the plane made by particles "p1", "p2" and "p3"
    #   using the potential "pot_dih"
    dihedral_handle = mx.Dihedral.create(pot_dih, p0, p1, p2, p3)
    all_dihedrals = mx.Universe.dihedrals()  # Get updated list of all dihedrals

A dihedral is rendered as a line joining the first and second particles, a
line joining the third and fourth particles, and a line joining the midpoint
of those two lines.
