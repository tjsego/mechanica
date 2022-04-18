import mechanica as mx
import numpy as np

mx.init(dim=[25., 25., 25.], cutoff=3, bc=mx.BOUNDARY_NONE, windowless=True)


class BlueType(mx.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = mx.Overdamped
    style = {'color': 'dodgerblue'}


class BigType(mx.ParticleType):
    mass = 1
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Blue = BlueType.get()
Big = BigType.get()

# simple harmonic potential to pull particles
pot = mx.Potential.harmonic(k=1, r0=0.1, max=3)

# make big cell in the middle
Big(mx.Universe.center)

# Big.style.visible = False

# create a uniform mesh of particles and bonds on the surface of a sphere
parts, bonds = mx.bind.sphere(pot, type=Blue, n=5, phi=(0.6 * np.pi, 0.8 * np.pi), radius=Big.radius + Blue.radius)

# run the model
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
