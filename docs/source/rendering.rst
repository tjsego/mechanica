.. _rendering:

Rendering and System Interaction
--------------------------------

Mechanica provides a number of methods to interact with the rendering
engine and host CPU via the ``system`` module (:class:`MxSystem` in C++).
The view of a simulation can be retrieved or set at any time during simulation,
including in :ref:`custom keyboard commands <events_input_driven>`, ::

    import mechanica as mx
    mx.init()

    # camera view parameters for storing and reloading
    view_center, view_rotation, view_zoom = None, None, None
    # key "s" stores a view; key "d" restores a stored view
    def do_key_actions(event):
        if event.key_name == "s":
            global view_center, view_rotation, view_zoom
            view_center = mx.system.cameraCenter()
            view_rotation = mx.system.cameraRotation()
            view_zoom = mx.system.cameraZoom()
        elif event.key_name == "d" and view_center is not None:
            mx.system.cameraMoveTo(view_center, view_rotation, view_zoom)

    mx.on_keypress(do_key_actions)

Simulation scripts can also define precise adjustments to the camera view
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

Basic information about a Mechanica installation can be retrieved on demand,
including information about the CPU, software compilation and available graphics
hardware, ::

    print('CPU info:', mx.system.cpu_info())
    print('Compilation info:', mx.system.compile_flags())
    print('OpenGL info:', mx.system.gl_info())

