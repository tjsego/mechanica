.. _style:

Style
------

All renderable objects in Mechanica have a ``style`` attribute. The style object
behaves like a container for a variety of style descriptors.
Each instance of an object with a ``style`` automatically inherits the style of
its type, which can then be individually manipulated. The ``style`` attribute
currently supports setting the color (:meth:`setColor`) and visibility
(:meth:`setVisible`) of its parent object.

Styling Particle Types in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ParticleType` has a special procedure for specifying the style of
a type as a class definition in Python. The ``style`` attribute can be defined
in Python as a dictionary with key-value pairs for particle type class
definitions. The color of a type can be specified with the key ``"color"``
and value of the name of a color as a string. The visibility of a type can be
specified with key ``"visible"`` and value of a Boolean. ::

    import mechanica as mx

    class MyParticleType(mx.ParticleType):
        style = {'color': 'CornflowerBlue', 'visible': False}

    my_particle_type = MyParticleType.get()
    my_particle = my_particle_type()
    my_particle.style.setVisible(True)
