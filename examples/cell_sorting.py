import mechanica as m
import numpy as np

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
m.init(dim=dim, cutoff=cutoff)


class AType(m.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = m.Overdamped
    style = {'color': 'red'}


A = AType.get()


class BType(m.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = m.Overdamped
    style = {'color': 'blue'}


B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = m.Potential.morse(d=3,   a=5, max=3)
pot_bb = m.Potential.morse(d=3,   a=5, max=3)
pot_ab = m.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
m.bind.types(pot_aa, A, A)
m.bind.types(pot_bb, B, B)
m.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.Force.random(0, 50)

# bind it just like any other force
m.bind.force(rforce, A)
m.bind.force(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count, 3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count, 3)) * 15 + 2.5:
    B(p)

# run the simulator
m.Simulator.run()








