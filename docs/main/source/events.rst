.. _events:

.. py:currentmodule:: mechanica

Events
-------

An *event* is a set of procedures that occurs when a condition is satisfied (triggered).
Mechanica provides a robust event system with an ever-increasing library of
built-in events, as well as support for defining fully customized events.
In Mechanica, the procedures that correspond to an event are specified in a
user-specified, custom function. Each type of built-in event corresponds to a
particular condition by which Mechanica will evaluate the custom function,
as well as to a particular set of simulation information that Mechanica will
provide to the custom function.

The custom function that performs the set of procedures of an event is called
the *invoke method*. Aside from the condition that corresponds to a particular built-in
event, the condition of each event can be further customized by also specifying a
*predicate method*, which is another custom function that, when evaluated, tells
Mechanica whether the event is triggered. Both invoke and predicate methods take as
argument an instance of a specialized class of a base class :py:attr:`Event`
(:class:`MxEvent` in C++). In C++, pointers to invoke and predicate methods can be
created using the template :class:`MxEventMethodT` defined in *MxEvent.h*, where the
template parameter is the class of the corresponding event. Invoke methods return
``1`` if an error occurred during evaluation, and otherwise ``0``. Predicate methods
return ``1`` if the event should occur, ``0`` if the event should not occur, and a
negative value if an error occurred.

Working with Events
^^^^^^^^^^^^^^^^^^^^

In the most basic (and also most robust) case, Mechanica provides a basic
:py:attr:`Event` class that, when created, is evaluated at every simulation step.
A :py:attr:`Event` instance has no built-in predicate method, and its invoke method is
evaluated at every simulation step unless a custom predicate method is provided.
As such, the :py:attr:`Event` class is the standard class for implementing custom
events for a particular model and simulation. An :py:attr:`Event` instance
can be created with the top-level method :func:`on_event`
(:func:`MxOnEvent` in C++). ::

    import mechanica as mx
    ...
    # Invoke method: destroy the first listed particle in the universe
    def destroy_first_invoke(event):
        particle = mx.Universe.particles()[0]
        particle.destroy()
        return 0

    mx.on_event(invoke_method=destroy_first_invoke)

In this example, a particle is destroyed at every simulation step, which could be
problematic in the case where no particles exist in the universe. Assigning a predicate
method to the event could solve such a problem that describes appropriate conditions
for the event to occur, ::

    # Predicate method: first particle is destroyed if there are more than ten particles
    def destroy_first_predicate(event):
        predicate = len(mx.Universe.particles()) > 10
        return int(predicate)

    mx.on_event(invoke_method=destroy_first_invoke,
                predicate_method=destroy_first_predicate)

Events that are called repeatedly can also be designated for removal from the event
system using the :py:attr:`Event` method :meth:`remove <MxEventBase.remove>` in an invoke
method. ::

    class SplittingType(mx.ParticleType):
        pass

    splitting_type = SplittingType.get()

    # Split once each step until 100 particles are created
    def split_to_onehundred(event):
        particles = splitting_type.items()
        num_particles = len(particles)
        if num_particles == 0:
            splitting_type()
        elif num_particles >= 100:
            event.remove()
        else:
            particles[0].split()
        return 0

    mx.on_event(invoke_method=split_to_onehundred)

Timed Events
^^^^^^^^^^^^^

The built-in event :py:attr:`TimeEvent` (:class:`MxTimeEvent` in C++) repeatedly
occurs with a prescribed period. By default, the period of evaluation is
approximately implemented as the first simulation time at which at an amount
of time at least as great as the period has elapsed since the last evaluation
of the event. :py:attr:`TimeEvent` instances can be created with the top-level
method :func:`on_time` (:func:`MxOnTimeEvent` in C++). ::

    def split_regular(event):
        splitting_type()
        return 0

    mx.on_time(invoke_method=split_regular, period=10.0)

The period of evaluation can also be implemented stochastically using the
optional keyword argument ``distribution``, which names a built-in distribution
by which Mechanica will generate the next time of evaluation from the event
period. Currently, Mechanica supports the Poisson distribution, which has
the name `"exponential"`. ::

    def split_random(event):
        splitting_type()
        return 0

    mx.on_time(invoke_method=split_random, period=10.0, distribution="exponential")

:py:attr:`TimeEvent` instances can also be generated for only a particular period
in simulation. The optional keyword argument ``start_time`` (default 0.0)
defines the first time in simulation when the event can occur, and the optional
keyword argument ``end_time`` (default forever) defines the last time in
simulation when the event can occur. ::

    def destroy_for_a_while(event):
        particles = splitting_type.items()
        if len(particles) > 0:
            particles[0].destroy()
        return 0

    mx.on_time(invoke_method=destroy_for_a_while, period=10.0,
               start_time=20.0, end_time=30.0)

Events with Particles
^^^^^^^^^^^^^^^^^^^^^^

Mechanica provides built-in events that operate on individual particles on
the basis of particle type. In addition to working with a custom invoke
method and optional predicate method, particle events select a particle
from a prescribed particle type. These event instances have the attributes
:attr:`targetType` and :attr:`targetParticle` that are set to the particle
type and particle that correspond to an event.

The :py:attr:`ParticleEvent` (:class:`MxParticleEvent`) is a particle event
that functions much the same as :py:attr:`Event`. A :py:attr:`ParticleEvent`
instance has an invoke method and optional predicate method, and is
evaluated at every simulation step. However, a :py:attr:`ParticleEvent`
instance also has an associated particle type and, on evaluation, an
associated particle. :py:attr:`ParticleEvent` instances can be created with
the top-level method :func:`on_particle` (:func:`MxOnParticleEvent` in C++). ::

    def split_selected(event):
        selected_particle = event.targetParticle
        selected_particle.split()
        return 0

    mx.on_particle(splitting_type, invoke_method=split_selected)

By default, a particle is randomly selected during the evaluation of a
particle event according to a uniform distribution. The largest particle
(*i.e.*, the cluster with the most constituent particles) can also be selected
using the optional keyword argument ``selector`` and passing ``"largest"``. ::

    def invoke_destroy_largest(event):
        event.targetParticle.destroy()
        return 0

    mx.on_particle(splitting_type, invoke_method=invoke_destroy_largest,
                   selector="largest")

The particle event :py:attr:`ParticleTimeEvent` (:class:`MxParticleTimeEvent` in C++)
functions is a combination of :py:attr:`TimeEvent` and :py:attr:`ParticleEvent`, and
can be created with the top-level method :func:`on_particletime`
(:func:`MxOnParticleTimeEvent` in C++) with all of the combined corresponding
arguments. ::

    def split_selected_later(event):
        event.targetParticle.split()
        return 0

    mx.on_particletime(splitting_type, period=10.0,
                       invoke_method=split_selected_later, start_time=20.0)

.. _events_input_driven:

Input-Driven Events
^^^^^^^^^^^^^^^^^^^^

Mechanica provides an event :py:attr:`KeyEvent` (:class:`MxKeyEvent` in C++) that
occurs each time a key on the keyboard is pressed. :py:attr:`KeyEvent` instances
do not support a custom predicate method. The name of the key that triggered
the event is available as the :py:attr:`KeyEvent` string attribute
:attr:`key_name <MxKeyEvent.key_name>`. One :py:attr:`KeyEvent` instance can be
created with the top-level method :func:`on_keypress`. In C++, an invoke method
can be assigned as a keyboard callback using the static method
:meth:`MxKeyEvent::addDelegate`. ::

    # key "d" destroys a particle; key "c" creates a particle
    def do_key_actions(event):
        if event.key_name == "d":
            particles = splitting_type.items()
            if len(particles) > 0:
                particles[0].destroy()
        elif event.key_name == "c":
            splitting_type()
        return 0

    mx.on_keypress(do_key_actions)
