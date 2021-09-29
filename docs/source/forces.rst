.. _forces:

Forces
-------

Forces cause objects to move. In addition to forces that result from various
processes (*e.g.*, interactions via a :ref:`potential <potentials>`),
Mechanica also supports modeling explicit forces using a suite of
built-in forces, as well as custom forces. An instance of any of the built-in
forces can be created using a static method on the :class:`Force` class
(:class:`MxForce` in C++), which can be :ref:`bound <binding>` to particles
by particle type.

Creating Forces
^^^^^^^^^^^^^^^^

:class:`Force` objects are created simply by calling one of the static methods
on the :class:`Force` class. For example, a random force can be created for
adding noise to the trajectory of particles, ::

    import mechanica as mx
    force = mx.Force.random(0.0, 1.0)

Custom forces can be created with the :class:`ConstantForce` class
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

A :class:`Force` instance can also be created by adding two existing
instances. Such operations can be arbitrarily performed to construct complicated
forces consisting of multiple constituent forces, ::

    force_noisy = mx.Force.random(0, 50)
    force_tstat = mx.Force.berenderson_tstat(10)
    force_noisy_tstat = force_noisy + force_tstat

.. note::

    Changes to constituent forces during simulation are reflected in forces
    that have been constructed from them using summation operations.

Built-in Forces
^^^^^^^^^^^^^^^^

Presently, the following built-in forces are supported, with corresponding
constructor method. For details on the parameters of each function, refer to the
:ref:`Mechanica API Reference <api_reference>`.

* Berendsen thermostat: Force.berendsen_tstat
* Friction: Force.friction
* Random: Force.random
