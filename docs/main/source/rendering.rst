.. _rendering:

Rendering and System Interaction
--------------------------------

Mechanica provides a number of methods to interact with the rendering
engine and host CPU via the ``system`` module (:class:`MxSystem` in C++).
Basic information about a Mechanica installation can be retrieved on demand,
including information about the CPU, software compilation and available graphics
hardware, ::

    import mechanica as mx
    mx.init()

    print('CPU info:', mx.system.cpu_info())
    print('Compilation info:', mx.system.compile_flags())
    print('OpenGL info:', mx.system.gl_info())

The ``system`` module provides rendering methods for customizing basic
visualization during simulation. Basic visualization customization combines
with specifications made using :ref:`Style <style>` objects, ::

    # Disable scene decorations
    mx.system.decorateScene(False)
    # Reduce shininess by 50%
    mx.system.setShininess(0.5 * mx.system.getShininess())
    # Move camera location
    mx.system.setLightDirection(mx.MxVector3f(1, 1, 1))
    # Save a screenshot with decorations and a white background
    mx.system.screenshot('mysim.jpg', decorate=True, bgcolor=[1, 1, 1])

Controlling the Camera
^^^^^^^^^^^^^^^^^^^^^^^

The view of a simulation can be retrieved or set at any time during simulation,
including in :ref:`custom keyboard commands <events_input_driven>`, ::

    # camera view parameters for storing and reloading
    view_center, view_rotation, view_zoom = None, None, None
    # key "s" stores a view; key "e" restores a stored view
    def do_key_actions(event):
        if event.key_name == "s":
            global view_center, view_rotation, view_zoom
            view_center = mx.system.cameraCenter()
            view_rotation = mx.system.cameraRotation()
            view_zoom = mx.system.cameraZoom()
        elif event.key_name == "e" and view_center is not None:
            mx.system.cameraMoveTo(view_center, view_rotation, view_zoom)

    mx.on_keypress(do_key_actions)

Mechanica also provides commands to perform precise adjustments to the camera view
in terms of rotations, zoom and camera position, ::

    from math import pi
    # Move to an isometric view (camera position, view center, upward axis)
    mx.system.cameraMoveTo(mx.MxVector3f(10, 10, 10), mx.MxVector3f(0, 0, 0), mx.MxVector3f(0, 0, 1))
    # Zoom in
    mx.system.cameraZoomBy(10)
    # Rotate about the z-axis
    mx.system.cameraRotateByEulerAngle(mx.MxVector3f(0, 0, pi / 2))
    # Show a preview
    mx.show()
    # Reset the camera after closing and then run
    mx.system.cameraReset()
    mx.run()

Creating and Controlling Clip Planes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Large and densly populated simulations can generate regions of space that are difficult to
inspect during simulation. For such cases, Mechanica supports introducing clip planes
to the visualization of simulation data. A clip plane divides the simulation domain by an imaginary
plane, one side of which is visualized, and the other side of which is not visualized.
Mechanica supports up to 8 clip planes at any given time in simulation.

A Mechanica simulation can be initialized in Python with one or more clip planes using the keyword
argument ``clip_planes`` in the :meth:`init` method. Clip planes in Python are specified in a list of
tuples (in C++, a string with the same syntax is passed), where each tuple specifies a clip plane.
Each tuple contains two elements: a three-element list specifying a point on the clip plane, and
a three-element list specifying the components of the normal vector of the plane, ::

    import mechanica as mx
    # Initialize with a clip plane at the center along the y-z plane
    mx.init(dim=[10, 10, 10], clip_planes=[([5, 5, 5], [1, 0, 0])])

Existing clip planes can be retrieved using the :class:`ClipPlanes` (:class:`MxClipPlanes` in C++)
interface, which provides :class:`ClipPlane` (:class:`MxClipPlane` in C++) objects for interacting
with clip planes during a simulation, ::

    # See how many clip planes we currently have
    print('Number of clip planes:', mx.ClipPlanes.len())  # Prints "1", from init
    # Get the clip plane created during initialization
    clip_plane0 = mx.ClipPlanes.item(0)                   # Returned object is a mx.ClipPlane

The :class:`ClipPlanes` interface also provides the ability to create new clip planes
at any time during a simulation, ::

    # Create a second clip plane at the center along the x-z plane
    clip_plane1 = mx.ClipPlanes.create(mx.Universe.center, mx.MxVector3f(0, 1, 0))

A :class:`ClipPlane` instance provides a live interface to its clip plane in the Mechanica rendering
engine, so that clip planes can be manipulated or destroyed at any time in simulation after
their creation, ::

    # Move the first clip plane to the origin and cut diagonally across the domain
    clip_plane0.setEquation(mx.Universe.origin(), mx.MxVector3f(1, 1, 1))
    # Remove the second clip plane
    clip_plane1.destroy()
    mx.run()

.. note:: Destroying a :class:`ClipPlane` can have downstream effects on the validity of
    other :class:`ClipPlane` instances. When a :class:`ClipPlane` instance is created, it
    refers to a clip plane by index from a list of clip planes in the rendering engine.
    If a clip plane is removed from the middle of the list of clip planes, then all instances
    after it in the list are shifted downward (like popping from a Python list). As such, all
    :class:`ClipPlane` instances that refer to downshifted clip planes have invalid reference
    indices. Invalid references can be repaired by decrementing their attribute :attr:`index`,
    though a more reliable approach is to always refer to clip planes using the
    :class:`ClipPlanes` static method :meth:`item` (*e.g.*, ``mx.ClipPlanes.item(1).destroy()``).
