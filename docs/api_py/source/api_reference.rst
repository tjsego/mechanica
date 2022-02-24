Mechanica Library Python API Reference
=======================================

.. module:: mechanica
   :platform: Linux, Windows
   :synopsis: Interactive, particle-based physics, chemistry and biology modeling and simulation environment

This is the API Reference page for the module: :mod:`mechanica`

.. moduleauthor:: T.J. Sego <tjsego@iu.edu>


.. autofunction:: init

.. autofunction:: run

.. autofunction:: irun

.. autofunction:: close

.. autofunction:: show

.. autofunction:: step

.. autofunction:: stop

.. autofunction:: start

.. autofunction:: random_points

.. autofunction:: points

.. autofunction:: getSeed

.. autofunction:: setSeed

.. autoclass:: mechanica.version
    :members:

.. data:: has_cuda

    :type: boolean

    Flag signifying whether CUDA support is installed.

.. include:: api_simulator.rst

.. include:: api_universe.rst

.. include:: api_boundary.rst

.. include:: api_constants.rst

.. include:: api_particle.rst

.. include:: api_particlelist.rst

.. include:: api_potential.rst

.. include:: api_forces.rst

.. include:: api_species.rst

.. include:: api_flux.rst

.. include:: api_bind.rst

.. include:: api_bonded.rst

.. include:: api_events.rst

.. include:: api_lattice.rst

.. include:: api_style.rst

.. include:: api_rendering.rst

.. include:: api_system.rst

.. include:: api_logger.rst

.. include:: api_types.rst

.. include:: api_io.rst

.. include:: api_cuda.rst
