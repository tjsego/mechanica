import mechanica as mx
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff)


class BeadType(mx.ParticleType):
    mass = 0.4
    radius = 0.2
    dynamics = mx.Overdamped


Bead = BeadType.get()

pot_bb = mx.Potential.coulomb(q=0.1, min=0.1, max=1.0)

# hamonic bond between particles
pot_bond = mx.Potential.harmonic(k=0.4, r0=0.2, min=0, max=2)

# angle bond potential
pot_ang = mx.Potential.harmonic_angle(k=0.01, theta0=0.85 * np.pi, tol=0.01)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_bb, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=0.1)

# bind it just like any other force
mx.bind.force(rforce, Bead)

# make a array of positions
xx = np.arange(0.2/2, dim[0] - 0.2/2, 0.2)

p = None                              # previous bead
bead = Bead([xx[0], 10., 10.0])       # current bead
bead_init = bead

for i in range(1, xx.size):
    n = Bead([xx[i], 10.0, 10.0])     # create a new bead particle
    mx.Bond.create(pot_bond, bead, n)  # create a bond between prev and current
    if i > 1:
        mx.Angle.create(pot_ang, p, bead, n)  # make an angle bond between prev, cur, next
    p = bead
    bead = n

mx.Bond.create(pot_bond, bead, bead_init)  # create a bond between first and last
mx.Angle.create(pot_ang, p, bead, bead_init)  # make an angle bond between prev, last, first

# run the simulator interactive
mx.run()
