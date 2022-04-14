import mechanica as mx
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff)


class BeadType(mx.ParticleType):
    mass = 1
    radius = 0.5
    dynamics = mx.Overdamped


Bead = BeadType.get()
pot = mx.Potential.glj(e=1)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=0.01)
mx.bind.force(rforce, Bead)


r = 0.8 * Bead.radius

positions = [mx.Universe.center + [x, 0, 0] for x in np.arange(-count * r + r, count * r + r, 2 * r)]


for p in positions:
    print("position: ", p.as_list())
    Bead(p)

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
