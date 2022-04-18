import mechanica as mx

mx.init(dim=[20., 20., 20.], cutoff=8, bc=mx.BOUNDARY_NONE, windowless=True)


class BeadType(mx.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = mx.Overdamped


Bead = BeadType.get()

# simple harmonic potential to pull particles
pot = mx.Potential.harmonic(k=1, r0=0.1, max = 3)

# make a ring of of 50 particles
pts = [p * 5 + mx.Universe.center for p in mx.points(mx.PointsType.Ring, 50)]

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
mx.bind.bonds(pot, beads, 1)

# run the model
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
