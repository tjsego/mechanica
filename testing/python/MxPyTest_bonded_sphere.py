import mechanica as mx
import numpy as np

mx.init(dim=[25., 25., 25.], cutoff=3, dt=0.005, bc=mx.BOUNDARY_NONE, windowless=True)


class BlueType(mx.ParticleType):
    mass = 10
    radius = 0.05
    dynamics = mx.Overdamped
    style = {'color': 'dodgerblue'}


Blue = BlueType.get()


class BigType(mx.ParticleType):
    mass = 10
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Big = BigType.get()

# simple harmonic potential to pull particles
h = mx.Potential.harmonic(k=200, r0=0.001, max=5)

# simple coulomb potential to maintain separation between particles
pb = mx.Potential.coulomb(q=0.01, min=0.01, max=3)

# potential between the small and big particles
pot = mx.Potential.glj(e=1, m=2, max=5)

Big(mx.Universe.center)

Big.style.visible = False

mx.bind.types(pot, Big, Blue)

mx.bind.types(pb, Blue, Blue)

parts, bonds = mx.bind.sphere(h, type=Blue, n=4, phi=[0.55 * np.pi, 1 * np.pi], radius=Big.radius + Blue.radius)

# run the model
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
