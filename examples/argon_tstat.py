import mechanica as mx
import numpy as np

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim = [10., 10., 10.]

# new simulator
mx.init(dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = mx.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)


# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class ArgonType(mx.ParticleType):
    mass = 39.4
    target_energy = 10000


Argon = ArgonType.get()

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, Argon, Argon)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = mx.Force.berenderson_tstat(10)

# bind it just like any other force
mx.bind.force(tstat, Argon)

size = 100

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 3))
velocities = np.random.normal(0, 0.1, size=(size,3))

for (pos, vel) in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos, vel)

# run the simulator interactive
mx.Simulator.run()
