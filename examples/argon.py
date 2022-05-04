import mechanica as mx
import numpy as np

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
mx.init(dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
pot = mx.Potential.lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, 1.0e-3)


# create a particle type
class ArgonType(mx.ParticleType):
    radius = 0.1
    mass = 39.4


# Register and get the particle type; registration always only occurs once
Argon = ArgonType.get()

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, Argon, Argon)

# create lots of particles in a uniform random cube
Argon.factory(positions=np.random.uniform(low=0, high=10, size=(10000, 3)))

# run the simulator interactive
mx.run()
