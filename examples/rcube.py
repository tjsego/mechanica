import mechanica as mx

mx.init(dim=[20., 20., 20.], cutoff=8, bc=mx.BOUNDARY_NONE)


class BeadType(mx.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = mx.Overdamped


class BlueType(mx.ParticleType):
    mass = 1
    radius = 0.1
    dynamics = mx.Overdamped
    style = {'color': 'blue'}


Bead = BeadType.get()
Blue = BlueType.get()

pot = mx.Potential.harmonic(k=1, r0=0.1, max = 3)

pts = [p * 18 + mx.Universe.center for p in mx.random_points(mx.PointsType.SolidCube, 10000)]

beads = [Bead(p) for p in pts]

mx.run()
