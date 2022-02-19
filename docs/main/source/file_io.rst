.. _file_io:

I/O Operations
---------------

Mechanica supports a number of operations associated with importing and exporting simulation and
simulation data. At any time during simulation, data can be archived during simulation for later
import, execution and analysis, or exported to common 3D model or `JSON <https://www.json.org/>`_
file formats for easy sharing of model objects and browsable three-dimensional simulation results
among colleagues, research groups and the broader scientific community. In general, I/O operations
are defined in the ``io`` module (:class:`MxIO` in C++). For detailed information on classes and
methods available in the ``io`` module, refer to the :ref:`Mechanica API Reference <api_io>`.

Loading and Saving a Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mechanica supports saving the state of a simulation to file at any time during simulation. Almost
all simulation data can be written to file in JSON format using
the ``io`` module command :meth:`toFile`, ::

    import mechanica as mx
    from os import path

    mx.init()

    # Simulation code here...

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to export
    mx.io.toFile(fp)                                                         # Export to file

Data files exported using :meth:`toFile` can be imported and used to initialize a simulation in
approximately the same state as when the exported file was generated. The path to a file containing
exported data can be passed directly to the keyword ``load_file`` of :meth:`init` when initializing
Mechanica, ::

    import mechanica as mx
    from os import path

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to import
    mx.init(load_file=fp)

Initialization from an imported simulation state is not limited to the data defined in the
imported file. Rather, Mechanica begins by initializing from whatever data is defined in the imported
file, and then resumes all other initializations in the typical way. For example, exported data includes
select simulator details like the timestep used by the simulator. However, passing an explicit timestep
to Mechanica during initialization will use the specified timestep, and not the one defined in the imported
file, ::

    import mechanica as mx
    from os import path

    fp = path.join(path.dirname(path.abspath(__file__)), 'fileexport.json')  # Path of file to import
    mx.init(load_file=fp, dt=0.005)

Furthermore, all Mechanica functionality concerning creating objects and interactions between them
are fully available after importing a simulation state, while (almost) all objects and processes
defined in the imported file are also available after initialization, ::

    # Get an instance of particle type named "ExportedType" from imported data
    Exported = mx.MxParticleType_FindFromName('ExportedType')
    # Print the number of imported ExportedType particles
    print('Number of imported particles:', len(Exported.items()))
    # Create a few more ExportedType particles
    [Exported() for _ in range(10)]

The versatility of Mechanica's approach to importing simulation data comes with the tradeoff of
that not all simulation data is conserved during import. Certain data used to identify
objects like particles and particle types are not necessarily
the same between an original simulation state and its state after import. For example, suppose that a
certain particle has an ``id`` attribute value of ``10`` at export. After import, the attribute value
for ``id`` is not guaranteed to again be ``10``. However, Mechanica provides mappings of simulation
state data from values in the original state at export to values in the current
simulation state after import, ::

    # Get the id of the particle that had an id of 20 at export
    id_part_20 = mx.io.mapImportParticleId(20)

.. note::

    All data import maps are available between initialization and the first simulation step,
    after which they are purged.

Not all features of Mechanica are (or even can be) written to file during export.
While rendering details of particles, particle types and all bond types are exported,
non-critical simulation details like camera view are not exported.
More importantly, features that rely on custom functions and callbacks
(*e.g.*, :ref:`custom potentials <potentials>`, :ref:`custom forces <forces>` and
:ref:`events <events>`) cannot be exported.
Whenever necessary, such features must be created and loaded into Mechanica in the same
way after import to reproduce the complete simulation state.
For a complete list of information exported by Mechanica feature, see :ref:`Appendix A <appendix_a>`.

3D Model Formats
^^^^^^^^^^^^^^^^^

Mechanica makes sharing 3D results simple. At any time during simulation execution, the state of
the simulation can be exported to a 3D model format as a mesh, ::

    fp_3df = path.join(path.dirname(path.abspath(__file__)), 'fileexport.stl')  # Path to export stl
    mx.io.toFile3DF(format="stl", filePath=fp_3df, pRefinements=2)              # Export stl mesh

Mechanica integrates the Open Asset Import Library (`Assimp <http://assimp.org/>`_) for working
with 3D model formats, and so
`all formats supported by Assimp <https://assimp-docs.readthedocs.io/en/latest/about/introduction.html>`_
are also supported by Mechanica.

Mechanica can also import mesh data in a 3D file and make it available for constructing
a simulation. The ``io`` method :meth:`fromFile3DF` returns a structure of mesh data as imported
from a 3D file, ::

    fp_mesh = path.join(path.dirname(path.abspath(__file__)), 'mesh.obj')  # Path of mesh to import
    io_struct = mx.io.fromFile3DF(fp_mesh)                                 # Import mesh
    # Print import summary
    print(io_struct.num_meshes, 'meshes')
    print(io_struct.num_faces, 'faces')
    print(io_struct.num_edges, 'edges')
    print(io_struct.num_nodes, 'nodes')
    print('Mesh centroid:', io_struct.centroid)

The :class:`Structure3DF` (:class:`MxStructure3DF` in C++) instance returned by :meth:`fromFile3DF`
contains all vertices, edges, faces and meshes imported from the 3D file, and provides a few useful
methods for using the mesh data in a simulation (`e.g.`, building a simulation from a mesh designed
in Blender), ::

    import math

    # Translate mesh centroid to center of universe
    io_struct.translateTo(mx.Universe.center)
    # Rotate 90 degress about X
    io_struct.rotate(mx.MxMatrix4f.rotationX(math.pi/2).rotation())
    # Double the size about the centroid
    io_struct.scale(2.0)

For example, particles can readily be constructed at each vertex of a mesh by simply iterating
over all vertices of the mesh, ::

    class VertexType(mx.ParticleType):
        """A type for particles built from mesh data"""
        pass

    Vertex = VertexType.get()
    # Create particles from mesh vertices
    for v in io_struct.vertices:
        Vertex(v.position)

Serializing Mechanica Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mechanica supports serialization of most objects using JSON strings for sharing individual model
objects. Any object that can be serialized has the methods :meth:`toString`, and its class has the static
method :meth:`fromString`. :meth:`toString` returns a JSON-formatted string of the state of the object,
which can be exported for sharing, ::

    # A Mechanica simulation written by Modeler A.
    import mechanica as mx
    from os import path
    mx.init()

    class ParticleTypeA(mx.ParticleType):
        """Awesome Mechanica particle"""

    A = ParticleTypeA.get()
    # Export the type to share with a friend
    fp = path.join(path.dirname(path.abspath(__file__)), 'ptypea.json')
    with open(fp, 'w') as f:
        f.write(A.toString())

The generated string can later be used by the :meth:`fromString` method of the class that generated the
string to recreate the object, ::

    # A Mechanica simulation written by Modeler B.
    import mechanica as mx
    from os import path
    mx.init()

    # Import a type shared by a friend
    fp = path.join(path.dirname(path.abspath(__file__)), 'ptypea.json')
    with open(fp, 'r') as f:
        A = mx.MxParticleType.fromString(f.read())

Mechanica provides built-in support in Python for pickling all objects that can be serialized.
All objects that support pickling can be seemlessly integrated into multithreading applications, ::

    from multiprocessing import Pool

    def energy_diff(bond):
        """Calculates the difference of the potential and dissociation energies of a bond"""
        return bond.dissociation_energy - bond.potential_energy

    # Calculate all bond energy differences in parallel
    with Pool(8) as p:
        energy_diffs = p.map(energy_diff, [bh.get() for bh in mx.Universe.bonds()])

All objects that can be pickled have the method :meth:`__reduce__` marked in the
documentation of their class in the :ref:`Mechanica API Reference <api_reference>`.

.. note:: Special care must be taken to account for that deserialized Mechanica objects are copies of
    their original object, and that the Mechanica engine is not available in separate processes. As such,
    calls to methods that require the engine in a spawned Python process will fail.
