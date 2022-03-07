"""
This example demonstrates basic programmatic usage of Mechanica camera controls.

*Warning*

Presentation of the simulation defined in this script may cause motion sickness.
"""
import mechanica as mx
from math import sin, pi

mx.init()


class AType(mx.ParticleType):
    radius = 0.1


A = AType.get()

# Create a simple oscillator
pot = mx.Potential.harmonic(k=100, r0=0.3)
disp = mx.MxVector3f(A.radius + 0.07, 0, 0)
p0 = A(mx.Universe.center - disp)
p1 = A(mx.Universe.center + disp)
mx.Bond.create(pot, p0, p1)

# Vary the camera view
move_ang = 0.025                 # Rate of camera rotation
move_zom = 0.1                   # Rate of camera zoom
cam_per = mx.Universe.dt * 1000  # Camera period
rot_idx = 1                      # Index of current camera axis of rotation

# Initialize camera position
pos_reset = mx.MxVector3f(0, 0, 0)
rot_reset = mx.MxQuaternionf(mx.MxVector3f(0.5495796799659729, 0.09131094068288803, 0.08799689263105392),
                             -0.8257609605789185)
zoom_reset = -2.875530958175659
mx.system.cameraMoveTo(pos_reset, rot_reset, zoom_reset)


def auto_cam_move(e):
    """Event callback to vary the camera"""
    cf = sin(2 * pi * mx.Universe.time / cam_per)
    angles = mx.MxVector3f(0.0)
    angles[rot_idx] = move_ang * cf
    zoom = - move_zom * cf
    mx.system.cameraRotateByEulerAngle(angles=angles)
    mx.system.cameraZoomBy(zoom)


def inc_cam_rotation_axis(e):
    """Event callback to change the camera axis of rotation"""
    global rot_idx
    rot_idx = rot_idx + 1 if rot_idx < 2 else 0


mx.on_time(period=mx.Universe.dt, invoke_method=auto_cam_move)
mx.on_time(period=cam_per, invoke_method=inc_cam_rotation_axis)

# Run the simulator
mx.run()

# Report the final camera view settings before exiting
print('Camera position:', mx.system.cameraCenter())
print('Camera rotation:', mx.system.cameraRotation())
print('Camera zoom:', mx.system.cameraZoom())
