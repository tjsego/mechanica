GPU-Accelerated Modules
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    This section of the Mechanica API is only available in CUDA-supported installations.


.. autoclass:: SimulatorCUDAConfig

.. autoclass:: MxSimulatorCUDAConfig

    .. autoproperty:: engine

        :type: MxEngineCUDAConfig

    .. autoproperty:: bonds

        :type: MxBondCUDAConfig


.. autoclass:: EngineCUDAConfig

.. autoclass:: MxEngineCUDAConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: setDevice

    .. automethod:: clearDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshPotentials

    .. automethod:: refreshFluxes

    .. automethod:: refreshBoundaryConditions

    .. automethod:: refresh

    .. automethod:: setSeed

    .. automethod:: getSeed


.. autoclass:: BondCUDAConfig

.. autoclass:: MxBondCUDAConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: setDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshBond

    .. automethod:: refreshBonds

    .. automethod:: refresh


.. autoclass:: AngleCUDAConfig

.. autoclass:: MxAngleCUDAConfig

    .. automethod:: onDevice

    .. automethod:: getDevice

    .. automethod:: toDevice

    .. automethod:: fromDevice

    .. automethod:: setBlocks

    .. automethod:: setThreads

    .. automethod:: refreshAngle

    .. automethod:: refreshAngles

    .. automethod:: refresh


.. autoclass:: cuda

.. autoclass:: MxCUDA

    .. automethod:: getDeviceName

    .. automethod:: getDeviceTotalMem

    .. automethod:: getDeviceAttribute

    .. automethod:: getNumDevices

    .. automethod:: getDevicePCIBusId

    .. automethod:: getCurrentDevice

    .. automethod:: maxThreadsPerBlock

    .. automethod:: maxBlockDimX

    .. automethod:: maxBlockDimY

    .. automethod:: maxBlockDimZ

    .. automethod:: maxGridDimX

    .. automethod:: maxGridDimY

    .. automethod:: maxGridDimZ

    .. automethod:: maxSharedMemPerBlock

    .. automethod:: maxTotalMemConst

    .. automethod:: warpSize

    .. automethod:: maxRegsPerBlock

    .. automethod:: clockRate

    .. automethod:: gpuOverlap

    .. automethod:: numMultiprocessors

    .. automethod:: kernelExecTimeout

    .. automethod:: computeModeDefault

    .. automethod:: computeModeProhibited

    .. automethod:: computeModeExclusive

    .. automethod:: PCIDeviceId

    .. automethod:: PCIDomainId

    .. automethod:: clockRateMem

    .. automethod:: globalMemBusWidth

    .. automethod:: L2CacheSize

    .. automethod:: maxThreadsPerMultiprocessor

    .. automethod:: computeCapabilityMajor

    .. automethod:: computeCapabilityMinor

    .. automethod:: L1CacheSupportGlobal

    .. automethod:: L1CacheSupportLocal

    .. automethod:: maxSharedMemPerMultiprocessor

    .. automethod:: maxRegsPerMultiprocessor

    .. automethod:: managedMem

    .. automethod:: multiGPUBoard

    .. automethod:: multiGPUBoardGroupId

    .. automethod:: test
