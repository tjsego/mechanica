.. _history:

History
========

Version 0.33.0
--------------
Minor features and bug fix release

* Improved performance of particle creation
* Upgrades to Python support for Windows
* Improved potential consistency, checks and performance
* Removed limitation on local populations in CUDA acceleration
* Added integration support in cmake projects
* Improved library interface
* Added C++ tests submodule
* Improved documentation
* Bug fixes

  * Disabled unreliable automated tests in pipelines conda build
  * Minor build fixes for gcc
  * Corrected pointers during fission
  * Fixed memory leak in I/O
  * Fixed incorrect install headers


Version 0.32.1
--------------
Patch release

* Added links to C API docs
* Reduced cost of docs build to support rendered notebooks online
* Mitigated bug related to pre-rendered notebooks with jpeg output
* Relaxed python library testing to mitigate random, erroneous errors on Azure for osx

Version 0.32.0
--------------
Minor features and bug fix release

* Adds C API
* Adds and integrates test suite
* Finishes all current features in lattice python module
* Updates Jupyter deployment and widgets
* Adds notebooks to distributed examples
* Adds rendered notebooks and gallery to docs examples
* Adds settable particle position, velocity and force
* Adds automatic destruction of angles and dihedrals connected to a destroyed particle
* Adds improved checks to particle handle and type
* Adds key modifiers to key events
* Adds support for multiple new screenshot file formats
* Splits language support and bindings into separate, self-contained modules
* Major cleanup of codebase
* Minor bug fixes

  * Fixes particle position details very close to boundaries
  * Removes redundant random force in Friction
  * Refactores Berendsen to correct typo
  * Fixes/improves argument handling in python particle construction
  * Fixes inventory tracking during particle destruction
  * Fixes cluster particle factory
  * Fixes visible types in particle become
  * Adds templates for universe grid

Version 0.31.0
--------------
Huge release, with completion of (hopefully) all features for Version 1.0.0

* Core features

  * Adds support for model specification in pure C++
  * Adds support for generating bindings in additional languages
  * Adds a formal event system, with callback capabilities in C++ and python
  * Implements universal naming of particle types: a registered type can now always be uniquely determined anywhere in a simulation
  * Adds "models" module for application-specific modeling features
  * Adds libSBML dependency
  * Removes carbon
  * Adds modular CUDA runtime support
  * Improves synchronization of SBML states and state vector dynamics
  * Adds simulation I/O based on JSON
  * Adds I/O support for 3D model format files
* Modeling features

  * Populates event system with events for

    * single events
    * time-dependent events
    * particle-dependent events
    * time- and particle-dependent events
  * Adds cell polarity model to models module
  * Adds "reset" boundary condition
  * Unifies bond interfaces
* Visualization / interactive features

  * Adds camera and basic visualization interfaces
  * Adds clip planes support for bonds rendering
  * Adds interactive interface for clip planes
  * Adds interface for random number generator
  * Adds interface to customize basic visualization (e.g., background color, scene decorations)
  * Adds screenshot interface
  * Adds rendering of space discretization
  * Adds visualization of view orientation
  * Adds keyboard commands for

    * pre-defined views
    * toggling scene decorations
    * toggling display of space discretization
* Documentation features

  * Adds documentation on (hopefully) all modeling and simulation features
  * Adds documentation on select back-end details
  * Adds automated C/C++ API documentation using Doxygen
  * Adds automated Python API documentation using Sphinx
* Build / distribution features

  * Adds automated local build using conda
  * Adds conda package recipe
  * Implements CI/CD using Azure

Version Alpha 1.0.30.4
----------------------
* fixed bugs in performance timers

Version Alpha 1.0.30.3
----------------------
* imporved logging, file logging
* bug fixes in determining if jupyter is running
* more logging info
* better error handling
* log performance counters
* expose performance counter in Python

Version Alpha 1.0.29.0
----------------------
* bug fixes in multi-threaded rendering from Python
* bug fixes in jupyter widget

Version Alpha 1.0.28.0
----------------------
* switch to Python 3.9 on Mac.
* disable jpeg on ARM

Version Alpha 1.0.27.0
----------------------
* Added massivly experimental support for ARM on M1 processor. Note, we have no
  way of testing this.

Version Alpha 1.0.26.0
----------------------
* Clipping Planes! users can specify arbitrary clipping planes with nice Python API.
* Forked Magnum Phong shader, we now have complete control over the shader.

Version Alpha 1.0.25.2
----------------------
* functions to detect if we are running in interactive terminal or
  jupyter notebook
* stubbed out new jupyter widget file.
* call jupyter widget init / run if we are running in jupyter notebook server.

Version Alpha 1.0.24.1
----------------------
* new coupling parameter between single body forces and chemical species on
  objects.

Version Alpha 1.0.23.1
----------------------
* bug fix in DPD potential between fluid and large objects
* new scaled DPD potential that enables fluid interaction between objects of
  varying size.

Version Alpha 1.0.22.1
----------------------
* lots of new bonds method, can iterate network connected by bonds
* fix in bind, to make bond to instances
* fix in parsing boundary conditions
* new 'reset' boundary condition for species attached to objects.

Version Alpha 1.0.21.1
----------------------
* major bug fix in moving particles to different cells if cell has both periodic
  and bounce-back boundary conditions.
* additional particle verify functions
* bug fix in virial calculation with DPD potentials

Version Alpha 1.0.20.1
----------------------
* new `universe.grid` method to get particles binned on grid locations
* improve error handling
* change some examples to use Morse potential
* doc updates
* force calculation bug fixes

Version Alpha 1.0.19.1
----------------------
* new Morse potential
* major bug fix in potential calculation
* add `reset_species` method on particle state vector
* species syntax parsing fixes, read boundary and init condition correctly
* lots of new view camera rotation functions in python api.

Version Alpha 1.0.18.1
----------------------
* generalized passive, consumer and producer fluxes
* better OpenGL info reporting, `gl_info()`, `egl_info()`
* enable boundary conditions on chemical speices, bug fix parsing init
  conditions
* use species boundary value to enable source / sinks
* source / sinks in example

Version Alpha 1.0.17.1
----------------------
* multi-threaded rendering fixes

Version Alpha 1.0.16.3
----------------------
* Logging, standardized all logging output, python api for setting log level.
* fix kinetic energy reporting
* synchronize gl contexts between GLFW and Magnum for multi-thread rendering

Version Alpha 1.0.16.2
----------------------
* initialize Mechanica either via m.init, m.Simulator, or m.simulator.init

Version Alpha 1.0.16.1
----------------------
* finally, completly expunged pybind11! pybind11 is finally GONE!
* context managment methods for multi-threaded headless rendering.
* universe.reset() method, clears objects
* set window title to script name
* add 'positions()', 'velocities()' and 'forces()' methods to particle list.
* universe.particles() is now a method, and returns a proper list

Version Alpha 1.0.15.6
----------------------
* bug fix with boundary condition constants

Version Alpha 1.0.15.5
----------------------
* bug fix with force calculation when distance too short: pic random separation
  vector of with minimal distance. Seems to work...
* better diagnostic messages
* added normal to boundary vectors

Version Alpha 1.0.15.4
----------------------
* generalized boundary conditions
* add potentials to boundary conditions
* velocity, free-slip, no-slip and periodic boundary conditions
* render updates, back face culling
* headless rendering, rendering without X11 using GLES on Linux
* generalized power potential
* much improved error handling, much more consistency
* particle list fixes
* Rigid Body Dynamics ! (only cuboids currently supported, but still rigid bodies)
* add potentials to rigid bodies
* python api rigid body updates
* rendering updates, more consistency, simplify
* rigid body particle interactions
* friction force
* more expunging pybind, soon, soon we will be rid of pybind.
* bond dissociation_energy (break strength)
* lattice initializer
* add bonds to lattice initliazer
* performance logging
* updates to dissapative particle dynamics forces
* enable adding DPD force to boundaries.
* generlized single body force (external force)
* fluid dynamics examples
* visco-elastic materials, with bond breaking
* single-body time-dependent force definitions in python

Version Alpha 1.0.15.2
----------------------
* initial dissapative particle dynamics
* doc constant force, dpd

Version Alpha 1.0.15.1
----------------------


Version Alpha 0.0.14.1
----------------------
* added convenience methods to get spherical and cartesian coords from lists
* updated example models
* update docs
* added plot function in examples to plot polar angle velocity.
* code cleanup

Version Alpha 0.0.14
--------------------
* All new FLUX / DIFFUSION / TRANSPORT, We've not got
  Transport-Dissipative-Dynamics working!!!
* secrete methods on particle to perform atomic secrete
* bug fixes in neighbor list, make sure neighbor don't contain the particle
* bug fixes in harmonic potential
* new overlapped sphere potential
* new potential plotting method, lots of nice improvements
* new examples
* update become to copy over species values
* lattice initializers
* add decay to flux
* detect hardware concurrency
* bug fix in Windows release-mode CPUID crash
* multi-threaded integration
* all new C++ thread pool, working on getting rid of OpenMP / pthreads
* event system bug fixes
* documentation updates



Version Alpha 0.0.13
--------------------
* preliminary SBML species per object support
* SBML parsing, create state vector per object
* cpuinfo to determine instruction set support
* neighbor list bug fixes
* improve and simplify events
* on_keypress event
* colormap support per SBML species

Version Alpha 0.0.12
--------------------
* free-slip boundary conditions
* rendering updates
* energy minimizer in initial condition generator
* updates to init condition code
* initial vertex model support


Version Alpha 0.0.11
--------------------
* new linear potential
* triagulated surface mesh generation for spheres, triangulate sphere
  surfaces with particles and bonds, returns the set.
* banded spherical mesh generation
* bug fixes in making particle list from python list
* points works with spherical geometry
* internal refactoring and updates
* Dynamic Bonds! can dynamically create and destory bonds
* lots of changes to deal with variable bond numbers
* rendering updates for dyanmic bonds
* particle init refactor
* added metrics (pressure, center of mass, etc...) to particle lists
* add properties and methods to Python bond API
* bond energy calcs avail in python
* bond_str and repr
* automatically delete delete bond if particle is deleted


Version Alpha 0.0.10-dev1
-------------------------
* bug fixes in bond pairwise search
* improved particle `__repr__`, `__str__`
* new `style` visible attribute to style to toggle visibility on any
  rendered object
* make show() work in command line mode
* internal changes for more consistent use of handles vs direct pointers
* `bind_pairwise` to search a particle list for pairs, and bind them with a
  bond.
* new `points` and `random_points` to generate position distributions
* spherical plot updates
* new `distance` method on particles
* implmement `become`  -- now allow dynamic type change
* big fixes in simulation start right away instead of wait for event
* basic bond rendering (still lines, will upgrade to cylinders in future
* render large particles with higher resolution
* new particle list composite structure, all particles returned
  to python in this new list type. fast low overhead list.
* major performance improvment, large object cutoff optimization
* numpy array conversion bug fix
* neighbor list for particles in range
* enumerate all particles of type with 'items()'
* new c++ <-> python type conversions, getting rid of pybind.
* better error handling, check space cells are compatible with periodic boundary
  conditions.
* add `start`, `stop`, `show`, etc. methods to top-level as convenience.
* fix ipython interaction with `show`, default is universe not running when showing
* enable single stepping and visualization with ipython
* enable start and stop with keyboard space bar.
* pressure tensor calculations, add to different objects.
* new `Universe.center` property
* better error handling in `Universe.bind`
* clean up of importing numpy
* expose periodic boundary conditions to python.
* periodic on individual axis.
* new metrics calculations, including center of mass, radius of gyration,
  centroid, moment of inertia
* new spherical coords method
* frozen particles
* add harmonic term to generalized Lennard-Jones 'glj' potential

Version Alpha 0.0.9-dev4
------------------------
* tweaks in example models
* more options (periodic, max distance) in simulator ctor
* add flags to potentials
* persistence time in random force
* frozen option for particles
* make glj also have harmonic potential
* in force eval, if distance is less than min, set eval force to value at min position.
* accept bound python methods for events

Version Alpha 0.0.9
-------------------
* all new cluster dynamics to create sub-cellular element models
* cluster splitting
* splitting via cleavage plane
* splitting via cleavage axis
* other splitting options
* new potential system to deal with cluster and non-cluster interactions
* revamped generalized Lennard-Jones (glj) potential
* new 'shifted' potential takes into account particle radius
* updated potential plotting
* more examples
* fixed major integrator bug

Version Alpha 0.0.8
-------------------
* explicit Bond and Angle objects
* new example apps
* new square well potential to model constrained particles
* bug fixes in potential
* thread count in Simulator init


Version Alpha 0.0.7
-------------------
* lots of changes related to running in Spyder.
* force windows of background process to forground
* detect if running in IPython connsole -- use different message loop
* fix re-entrancy bugs in ipython message loop.
* Spyder on Windows tested.

Version Alpha 0.0.6
-------------------
* lots of changes to simulation running / showing windows / closing windows, etc..
* documentation updates

Version Alpha 0.0.5 Dev 1
-------------------------

* Add documentation to event handlers, and example programs
* fix bugs in creating event events
* add version info to build system and make available as API.


Version Alpha 0.0.4 Dev 1
-------------------------
* All new particle rendering based on instanced meshes. Rendering quality is
  dramatically improved. Now in a position to do all sorts of discrete elements
  like ellipsoids, bonds, rigid particles, etc...
* Implement NOMStyle objects. This is essentially the CSS model, but for 3D
  applications. Each object has a 'style' property that's a collection of all
  sorts of style attributes. The renderer looks at the current object, and chain
  of parent objects to find style attributes. Basically the CSS approach.
* More demo applications.
* Memory bugs resolved.

Version Alpha 0.0.3 Dev 1
-------------------------
* Windows Build!
* lots of portability updates
* some memleak fixes

Version Alpha 0.0.2 Dev 5
-------------------------

* lots of new documentation
* reorganize utility stuff to utily file
* add performance timing info to particle engine
* add examples (multi-size particles, random force, epiboly,
  events with creation, destruction, mitosis, ...)
* new dynamics options, include both Newtonian (Velocity-Verlet) and
  over-damped.
* new defaults to set space cell size, better threading
* New explicit bond object
* add creation time / age to particle
* particle fission (mitosis) method (simple)
* clean up potential flags
* harmonic potential
* new reactive potential to trigger (partial implementation)
* random points function to create points for geometric regions
* prime number generator
* Fixed major bug in cell pair force calculation (was in wrong direction)
* major bug fix in not making sure potential distance does not go past end of
  interpolation segments.
* new random force
* new soft-sphere interaction potential
* add radius to particle type def
* update renderer to draw different sized particles
* add number of space cells to simulator constructor
* configurable dynamics (Newtonian, Over-damped), more to come
  particle delete functionality, and fix particle events
* examples bind events to destroy, creation and mitosis methods
* new event model

Version Alpha 0.0.1 Dev 3
-------------------------

* Refactoring of Particle python meta-types, simpler and cleaner
* Upgrade to GLFW 3.3
* New single body generalized force system
* Berendsen thermostat as first example single body generalized forces
* Per-type thermostat
* Arc-ball user interaction
* Simplify and eliminate redundancy between C++ and Python apps.


Version Alpha 0.0.1 Dev 2
-------------------------
* First public release
