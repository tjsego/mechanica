import mechanica as m

m.init(dim=[20., 20., 20.], cutoff=8, bc=m.BOUNDARY_NONE)


class BeadType(m.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped


Bead = BeadType.get()

# simple harmonic potential to pull particles
pot = m.Potential.harmonic(k=1, r0=0.1, max = 3)

# make a ring of of 50 particles
pts = [p * 5 + m.Universe.center for p in m.points(m.PointsType.Ring, 50)]

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
m.bind.bonds(pot, beads, 1)

# run the model
m.run()
