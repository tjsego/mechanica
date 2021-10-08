Particles and Clusters
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Particle
.. autoclass:: ParticleHandle


.. autoclass:: MxParticle

    .. automethod:: py_particle


.. autoclass:: MxParticleHandle

    .. autoproperty:: charge

    .. autoproperty:: mass

    .. autoproperty:: frozen

    .. autoproperty:: frozen_x

    .. autoproperty:: frozen_y

    .. autoproperty:: frozen_z

    .. autoproperty:: style

    .. autoproperty:: age

    .. autoproperty:: radius

    .. autoproperty:: name

    .. autoproperty:: position

    .. autoproperty:: velocity

    .. autoproperty:: force

    .. autoproperty:: id

    .. autoproperty:: type_id

    .. autoproperty:: species

    .. autoproperty:: bonds

    .. autoproperty:: angles

    .. autoproperty:: dihedrals

    .. automethod:: part

    .. automethod:: type

    .. automethod:: split

    .. automethod:: destroy

    .. automethod:: sphericalPosition

    .. automethod:: virial

    .. automethod:: become

    .. automethod:: neighbors

    .. automethod:: getBondedNeighbors

    .. automethod:: distance

    .. automethod:: to_cluster


.. autoclass:: MxParticleType

    .. autoproperty:: frozen

    .. autoproperty:: frozen_x

    .. autoproperty:: frozen_y

    .. autoproperty:: frozen_z

    .. autoproperty:: temperature

    .. autoproperty:: target_temperature

    .. autoattribute:: id

    .. autoattribute:: mass

    .. autoattribute:: charge

    .. autoattribute:: radius

    .. autoattribute:: kinetic_energy

    .. autoattribute:: potential_energy

    .. autoattribute:: target_energy

    .. autoattribute:: minimum_radius

    .. autoattribute:: dynamics

    .. autoattribute:: name

    .. autoattribute:: style

    .. autoattribute:: species

    .. automethod:: particle

    .. automethod:: particleTypeIds

    .. automethod:: __call__

        Alias of :meth:`_call`

    .. automethod:: _call

    .. automethod:: newType

    .. automethod:: registerType

    .. automethod:: on_register

    .. automethod:: isRegistered

    .. automethod:: get

    .. automethod:: items


.. autoclass:: Cluster
.. autoclass:: ClusterHandle


.. autoclass:: MxCluster
    :show-inheritance:


.. autoclass:: MxClusterParticleHandle
    :show-inheritance:

    .. autoproperty:: radius_of_gyration

    .. autoproperty:: center_of_mass

    .. autoproperty:: centroid

    .. autoproperty:: moment_of_inertia

    .. automethod:: __call__

        Alias of :meth:`_call`

    .. automethod:: _call

    .. automethod:: split


.. autoclass:: MxClusterParticleType
    :show-inheritance:

    .. automethod:: hasType

    .. automethod:: get
