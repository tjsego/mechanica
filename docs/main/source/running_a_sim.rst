.. _running_a_sim:

.. py:currentmodule:: mechanica

Running a Simulation
---------------------

In most cases, running a simulation is as simple as initializing the simulator
with the module :func:`init` function (:func:`MxSimulator_init`
or :func:`MxSimulator_initC` in C++), building the physical model, and running
the simulation by calling the module :func:`run` method (or :func:`irun` for
interactive mode in Python, or :func:`MxSimulator::run` method in C++).
However, Mechanica provides many options for controlling the simulation
during execution.

A running simulation has three key components: interacting with the operating
system, handling events (both model-specific events and user input), and integrating the
model in time (time stepping). Whenever the :func:`run` (or :func:`irun` in Python)
is invoked, it automatically start time stepping the simulation. However, Mechanica
provides additional methods to finely control displaying, time stepping and stopping
a simulation. For example, calling the module method :func:`show`
(:func:`MxSimulator::show` in C++) only displays the application window and starts the
internals of Mechanica without performing any time stepping.
The module :func:`start`, :func:`step`, :func:`stop` methods
(:meth:`MxUniverse::start`, :meth:`MxUniverse::step`, :meth:`MxUniverse::stop` in C++)
start the :ref:`universe <mechanica_universe>` time evolution,
perform a single time step, and stop the time evolution, respectively.
If the universe is stopped, the :meth:`Universe.start <MxUniverse.start>` method can be
called to continue where the universe was stopped. All methods to build and manipulate
the universe are available either with the universe stopped or running.

When passing no arguments to :func:`run` (when passing a negative value to
:func:`MxSimulator::run` in C++), the main window opens, and the simulation runs and
only returns when the window closes. When an argument is passed, the value is understood
as the simulation time at which the simulation should stop, at which point the windw closes.
With :func:`irun` in IPython, the main window opens, and the simulation runs and
responds to user interactions with it and its objects in real time while it runs.

For convenience, all simulation control methods are aliased as top-level methods in Python, ::

    import mechanica as mx  # import the package
    mx.init()               # initialize the simulator
    # create the model here
    ...
    mx.irun()               # run in interactive mode (only for ipython console)
    mx.run()                # display the window and run
    mx.close()              # close the main window
    mx.show()               # display the window
    mx.step()               # time steps the simulation
    mx.stop()               # stops the simulation

In C++, the simulator and universe can both be easily accessed to use the same methods,

.. code-block:: cpp

    #include <MxSimulator.h>
    #include <MxUniverse.h>

    MxSimulator_Config config;
    MxSimulator_initC(config);  // initialize the simulator
    // create the model here
    ...
    MxSimulator *sim = MxSimulator::get();
    MxUniverse *universe = getUniverse();
    sim->run()                  // display the window and run
    sim->close()                // close the main window
    sim->show()                 // display the window
    universe->step()            // time steps the simulation
    universe->stop()            // stops the simulation

.. _running_a_sim_windowless:

Running Windowless
^^^^^^^^^^^^^^^^^^^

Many applications like massively-parallel execution of lots of simulations
require running Mechanica without real-time rendering and interactivity, where
Mechanica can execute simulations hundreds to thousands of times faster.
Mechanica supports such an execution mode, called `Windowless`, in which case
all Mechanica functionality is the same, except that Mechanica does no rendering
except when instructed to do so in the instructions of a scripted simulation.

Mechanica can be informed that a simulation should be executed in Windowless mode
during initialization with the keyword argument ``windowless``, ::

    mx.init(windowless=True)

Execution of a simulation occurs through the module method :func:`step` (rather than
:func:`run`), where each call executes one simulation step, ::

    num_steps = int(1E6)  # Number of steps to execute
    for step_num in range(num_steps):
        mx.step()

Reproducible Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^

Some features of Mechanica are stochastic (*e.g.*, random :ref:`forces <forces>`).
Mechanica uses a pseudo-random number generator to implement stochasticity.
By default, Mechanica generates a different stream of random numbers on each
execution of a simulation. However, in cases where results from a simulation with
stochasticity need to be reproduced (*e.g.*, when :ref:`sharing results <file_io>`),
Mechanica can use the same stream of random numbers when given the seed of the
pseudo-random number generator. Mechanica accepts specification of the seed during
initialization with the keyword argument ``seed``, as well as at any time during
simulation, ::

    mx.init(seed=1)               # Set the seed during initialization...
    mx.setSeed(mx.getSeed() + 1)  # ... or after initialization.
