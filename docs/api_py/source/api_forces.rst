Forces
-------

.. currentmodule:: mechanica

.. autoclass:: Force

.. autoclass:: MxForce

    .. automethod:: bind_species

    .. automethod:: berendsen_tstat

    .. automethod:: random

    .. automethod:: friction

    .. automethod:: toString

    .. automethod:: fromString

    .. automethod:: __reduce__

.. autoclass:: ConstantForce

.. autoclass:: MxConstantForce
    :show-inheritance:

    .. autoproperty:: value

    .. autoproperty:: period

    .. automethod:: fromForce

.. autoclass:: MxConstantForcePy
    :show-inheritance:

    .. automethod:: fromForce

.. autoclass:: ForceSum

.. autoclass:: MxForceSum
    :show-inheritance:

    .. autoattribute:: f1

    .. autoattribute:: f2

    .. automethod:: fromForce

.. autoclass:: Berendsen
    :show-inheritance:

    .. autoattribute:: itau

    .. automethod:: fromForce

.. autoclass:: Gaussian
    :show-inheritance:

    .. autoattribute:: std

    .. autoattribute:: mean

    .. autoattribute:: durration_steps

    .. automethod:: fromForce

.. autoclass:: Friction
    :show-inheritance:

    .. autoattribute:: coef

    .. autoattribute:: std

    .. autoattribute:: mean

    .. autoattribute:: durration_steps

    .. automethod:: fromForce
