Events
-------

.. currentmodule:: mechanica

.. autoclass:: MxEventBase

    .. autoattribute:: last_fired

    .. autoattribute:: times_fired

    .. automethod:: remove

.. autoclass:: Event

.. autoclass:: MxEvent
    :show-inheritance:

.. autoclass:: MxEventPy
    :show-inheritance:

.. autofunction:: on_event


.. autoclass:: TimeEvent

.. autoclass:: MxTimeEventPy
    :show-inheritance:

    .. autoattribute:: next_time

    .. autoattribute:: period

    .. autoattribute:: start_time

    .. autoattribute:: end_time

.. autofunction:: on_time


.. autoclass:: ParticleEvent

.. autoclass:: MxParticleEvent
    :show-inheritance:

    .. autoattribute:: targetType

    .. autoattribute:: targetParticle

.. autoclass:: MxParticleEventPy
    :show-inheritance:

.. autofunction:: on_particle


.. autoclass:: ParticleTimeEvent

.. autoclass:: MxParticleTimeEvent
    :show-inheritance:

    .. autoattribute:: targetType

    .. autoattribute:: targetParticle

    .. autoattribute:: next_time

    .. autoattribute:: period

    .. autoattribute:: start_time

    .. autoattribute:: end_time

.. autoclass:: MxParticleTimeEventPy
    :show-inheritance:

.. autofunction:: on_particletime


.. autoclass:: KeyEvent

.. autoclass:: MxKeyEvent

    .. autoproperty:: key_name

    .. autoproperty:: key_alt

    .. autoproperty:: key_ctrl

    .. autoproperty:: key_shift

.. autofunction:: on_keypress
