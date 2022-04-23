import mechanica as mx
import numpy as np

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
mx.init(dim=dim, windowless=True,
        cells=[5, 5, 5],
        cutoff=1.0)

# create a potential representing a 12-6 Lennard-Jones potential
pot = mx.Potential.lennard_jones_12_6(0.275, 1.0, 9.5075e-06, 6.1545e-03, 1.0e-3)


# create a particle type
class ArgonType(mx.ParticleType):
    radius = 0.1
    mass = 39.4


# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, Argon, Argon)

# uniform cube
[Argon() for _ in range(2500)]

# run the simulator
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
