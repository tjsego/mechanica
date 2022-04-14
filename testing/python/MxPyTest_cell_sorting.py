import mechanica as mx
import numpy as np

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff)


class AType(mx.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = mx.Overdamped
    style = {'color': 'red'}


A = AType.get()


class BType(mx.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = mx.Overdamped
    style = {'color': 'blue'}


B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = mx.Potential.morse(d=3, a=5, max=3)
pot_bb = mx.Potential.morse(d=3, a=5, max=3)
pot_ab = mx.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_aa, A, A)
mx.bind.types(pot_bb, B, B)
mx.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=50)

# bind it just like any other force
mx.bind.force(rforce, A)
mx.bind.force(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count, 3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count, 3)) * 15 + 2.5:
    B(p)

# run the simulator
mx.step(20*mx.Universe.dt)


def test_pass():
    pass
