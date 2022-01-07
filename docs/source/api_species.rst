Reactions and Species
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Species

.. autoclass:: MxSpecies

    .. automethod:: __str__

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

    .. autoproperty:: id

    .. autoproperty:: name

    .. autoproperty:: species_type

    .. autoproperty:: compartment

    .. autoproperty:: initial_amount

    .. autoproperty:: initial_concentration

    .. autoproperty:: substance_units

    .. autoproperty:: spatial_size_units

    .. autoproperty:: units

    .. autoproperty:: has_only_substance_units

    .. autoproperty:: boundary_condition

    .. autoproperty:: charge

    .. autoproperty:: constant

    .. autoproperty:: conversion_factor


.. autoclass:: SpeciesValue

.. autoclass:: MxSpeciesValue

    .. autoproperty:: boundary_condition

    .. automethod:: initial_amount

    .. automethod:: initial_concentration

    .. automethod:: constant

    .. automethod:: secrete


.. autoclass:: SpeciesList

.. autoclass:: MxSpeciesList

    .. automethod:: __str__

    .. automethod:: __len__

    .. automethod:: __getattr__

    .. automethod:: __setattr__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: item

    .. automethod:: insert

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__


.. autoclass:: StateVector

.. autoclass:: MxStateVector

    .. automethod:: __str__

    .. automethod:: __len__

    .. automethod:: __getattr__

    .. automethod:: __setattr__

    .. automethod:: __getitem__

    .. automethod:: __setitem__

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

    .. autoattribute:: species
