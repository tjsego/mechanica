import mechanica as mx


mx.init(dim=[25., 25., 25.], dt=0.0005, cutoff=3, bc=mx.BOUNDARY_NONE)


class GreenType(mx.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = mx.Overdamped
    style = {'color': 'mediumseagreen'}


class BigType(mx.ParticleType):
    mass = 10
    radius = 8
    frozen = True
    style = {'color': 'orange'}


Green = GreenType.get()
Big = BigType.get()

# simple harmonic potential to pull particles
pot = mx.Potential.harmonic(k=1, r0=0.1, max=3)

# potentials between green and big objects.
pot_yc = mx.Potential.glj(e=1, r0=1, m=3, min=0.01)
pot_cc = mx.Potential.glj(e=0.0001, r0=0.1, m=3, min=0.005, max=2)

# random points on surface of a sphere
pts = [p * (Green.radius + Big.radius) + mx.Universe.center for p in mx.random_points(mx.PointsType.Sphere, 10000)]

# make the big particle at the middle
Big(mx.Universe.center)

# constuct a particle for each position, make
# a list of particles
beads = [Green(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
# mx.bind.bonds(pot, beads, 0.7)

rforce = mx.Force.random(0, 0.01, duration=0.1)

# hook up the potentials
# mx.bind.force(rforce, Green)
mx.bind.types(pot_yc, Big, Green)
mx.bind.types(pot_cc, Green, Green)

mx.bind.bonds(pot, [p for p in mx.Universe.particles() if p.position[1] < mx.Universe.center[1]], 1)

# run the model
mx.run()
