.. _api_cell_polarity:

Cell Polarity
^^^^^^^^^^^^^^

.. module:: mechanica.models.center.cell_polarity

This is the API Reference page for the module: :mod:`mechanica.models.center.cell_polarity`.
For details on the mathematics and modeling concepts, see the
:ref:`Cell Polarity Module Documentation <cell_polarity>`.

.. autoclass:: CellPolarity

.. autoclass:: _CellPolarity

    .. automethod:: getVectorAB

    .. automethod:: getVectorPCP

    .. automethod:: setVectorAB

    .. automethod:: setVectorPCP

    .. automethod:: registerParticle

    .. automethod:: unregister

    .. automethod:: registerType

    .. automethod:: getInitMode

    .. automethod:: setInitMode

    .. automethod:: getInitPolarAB

    .. automethod:: setInitPolarAB

    .. automethod:: getInitPolarPCP

    .. automethod:: setInitPolarPCP

    .. automethod:: forcePersistent

    .. automethod:: setDrawVectors

    .. automethod:: setArrowColors

    .. automethod:: setArrowScale

    .. automethod:: setArrowLength

    .. automethod:: load

    .. automethod:: potentialContact

.. autoclass:: MxCellPolarityPotentialContact
    :show-inheritance:

    .. autoattribute:: couplingFlat

    .. autoattribute:: couplingOrtho

    .. autoattribute:: couplingLateral

    .. autoattribute:: distanceCoeff

    .. autoattribute:: cType

    .. autoattribute:: mag

    .. autoattribute:: rate

    .. autoattribute:: bendingCoeff

.. autoclass:: PolarityForcePersistent
    :show-inheritance:

    .. autoattribute:: sensAB

    .. autoattribute:: sensPCP
