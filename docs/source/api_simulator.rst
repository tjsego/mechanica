Simulator
^^^^^^^^^^

.. autoclass:: Simulator

.. autoclass:: MxSimulator

    .. automethod:: get

    .. automethod:: run

    .. automethod:: show

    .. automethod:: close

    .. staticmethod:: getCUDAConfig()

        Gets the Simulator CUDA runtime control interface.

        Only available in CUDA-supported installations.

        :rtype: MxSimulatorCUDAConfig

.. autoclass:: MxSimulatorPy
    :show-inheritance:

    .. automethod:: irun
