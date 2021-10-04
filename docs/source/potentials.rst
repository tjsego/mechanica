.. _potentials:

Potentials
-----------

One of the main goals of Mechanica is to enable users to rapidly develop and
explore empirical or phenomenological models of active and biological matter in
the 100nm to multiple cm range. Supporting modeling and simulation in these
ranges requires a good deal of flexibility to create and calibrate potential
functions to model material rheology and particle interactions.

Mechanica provides a wide range of potentials in the :class:`Potential` class
(:class:`MxPotential` in C++). Any of the built-in potential functions
can be created as objects in a simulation using a static method on the
:class:`Potential` class, which can be :ref:`bound <binding>` to pairs and
groups of particles to implement models of interactions.

Creating, Plotting and Exploring Potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`Potential` objects are created simply by calling one of the
static methods on the :class:`Potential` class. In Python, :class:`Potential`
objects conveniently have a :meth:`plot` method that displays a ``matplotlib``
plot. For example, while working with the built-in
Generalized Lennard-Jones potential, ::

    import mechanica as mx
    pot = mx.Potential.glj(10)
    pot.plot()

results in 

.. image:: glj_plot.png
    :alt: usage
    :width: 300px
    :class: sphx-glr-single-img

A :class:`Potential` instance can also be created by adding two existing
instances. Such operations can be arbitrarily performed to construct complicated
potentials consisting of multiple constituent potentials, ::

    pot_charged = mx.Potential.coulomb(q=1)
    pot_fluid = mx.Potential.dpd(alpha=0.3, gamma=1, sigma=1, cutoff=0.6)
    pot_charged_fluid = pot_charged + pot_fluid

.. note::

    Changes to constituent potentials during simulation are reflected in potentials
    that have been constructed from them using summation operations.

Built-in Potentials
^^^^^^^^^^^^^^^^^^^^

Presently, the following built-in potential functions are supported, with corresponding
constructor method. For details on the parameters of each function, refer to the
:ref:`Mechanica API Reference <api_reference>`.

* 12-6 Lennard-Jones: Potential.lennard_jones_12_6
* 12-6 Lennard-Jones with shifted Coulomb: Potential.lennard_jones_12_6_coulomb
* Coulomb: Potential.coulomb
* Dissipative particle dynamics: Potential.dpd
* Ewald (real-space): Potential.ewald
* Generalized Lennard-Jones: Potential.glj
* Harmonic: Potential.harmonic
* Harmonic angle: Potential.harmonic_angle
* Harmonic dihedral: Potential.harmonic_dihedral
* Cosine dihedral: Potential.cosine_dihedral
* Linear: Potential.linear
* Morse: Potential.morse
* Overlapping sphere: Potential.overlapping_sphere
* Power: Potential.power
* Soft sphere: Potential.soft_sphere
* Well: Potential.well
