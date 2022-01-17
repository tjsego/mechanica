"""
This example demonstrates select usage of basic visualization customization
"""

import mechanica as mx
from math import sin, cos, pi

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

# Vary some basic visualization periodically during simulation


def vary_colors(e):
    rate = 2 * pi * mx.Universe.time / 1.
    sf = (sin(rate) + 1) / 2.
    sf2 = (sin(2 * rate) + 1) / 2.
    cf = (cos(rate) + 1) / 2.

    mx.system.setGridColor(mx.MxVector3f(sf, 0, sf2))
    mx.system.setSceneBoxColor(mx.MxVector3f(sf2, sf, 0))
    mx.system.setShininess(1000 * sf + 10)
    mx.system.setLightDirection(mx.MxVector3f(3 * (2 * sf - 1), 3 * (2 * cf - 1), 2.))
    mx.system.setLightColor(mx.MxVector3f((sf + 1.) * 0.5, (sf + 1.) * 0.5, (sf + 1.) * 0.5))
    mx.system.setAmbientColor(mx.MxVector3f(sf, sf, sf))


mx.on_time(period=mx.Universe.dt, invoke_method=vary_colors)

# Run the simulator
mx.run()
