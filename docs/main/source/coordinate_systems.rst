.. _coordinate_systems:

.. py:currentmodule:: mechanica

Coordinate Systems
------------------

The :ref:`Mechanica universe <mechanica_universe>` uses a Cartesian
coordinate system to describe the position of objects in the
universe. For spherical coordinates, Mechanica uses the spherical
coordinates :math:`(r, \theta, \phi)` as often used in mathematics:
radial distance :math:`r`, azimuthal angle :math:`\theta`,
and polar angle :math:`\phi`.

.. figure:: spherical.png
    :width: 400px
    :align: center
    :alt: alternate text
    :figclass: align-center

The cartesian coordinates of every particle are accessible
via the particle property :py:attr:`position <MxParticleHandle.position>` ::

    p = MyParticleType.items()[0]
    print('Particle position:', p.position)

Spherical coordinates of a particle can be accessed with respect to
the frame of the universe, to the position of another particle, or to
the position of an arbitrary point, using the particle method
:py:meth:`sphericalPosition <MxParticleHandle.sphericalPosition>` ::

    # Report some spherical coordinates in various frames of particle "p"
    print('Spherical coordinates (global)  :', p.sphericalPosition())
    print('Spherical coordinates (particle):', p.sphericalPosition(particle=p_other))
    print('Spherical coordinates (point)   :', p.sphericalPosition(origin=[0.0, 1.0, 2.0])