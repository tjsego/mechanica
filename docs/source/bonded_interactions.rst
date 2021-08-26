.. _bonded_interactions:

Bonded Interactions
--------------------

A bonded interaction is an interaction due to a *bond* between
a group of particles. A bond describes an interaction exclusively
between the group of particles using a :ref:`potential <potentials>`.
Mechanica currently supports the standard
:ref:`bond <bonded_interactions:Bonds>` and bond-like
:ref:`angle <bonded_interactions:Angles>`.

Bonds
^^^^^^

.. image:: bond.png
   :alt: Left floating image
   :class: with-shadow float-left
   :height: 125px

A bond describes an interaction between two particles in terms
of the distance between the two particles. A bond can be
created using the method :meth:`create` on the class
:class:`Bond` (:class:`MxBond` in C++), which returns a handle
to the newly created bond. A bond can be manually destroyed
using the :class:`Bond` method :meth:`destroy`.

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between particles "p0" and "p1" using the potential "pot_bond"
    bond_handle = mx.Bond.create(pot_bond, p0, p1)

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
method :meth:`destroy`.

.. rst-class::  clear-both

.. code-block:: python

    # Create a bond between particles "p0" and "p2" w.r.t.
    #   particle "p1" using the potential "pot_ang"
    bond_handle = mx.Bond.create(pot_ang, p0, p1, p2)
