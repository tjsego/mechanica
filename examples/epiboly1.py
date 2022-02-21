import mechanica as mx

# potential cutoff distance
cutoff = 0.5

count = 3000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff)


class YolkType(mx.ParticleType):
    mass = 500000
    radius = 3


class CellType(mx.ParticleType):
    mass = 5
    radius = 0.2
    target_temperature = 0
    dynamics = mx.Overdamped


Cell = CellType.get()
Yolk = YolkType.get()

pot_bs = mx.Potential.soft_sphere(kappa=5, epsilon=20, r0=2.9, eta=3, tol=0.1, min=0, max=9)
pot_ss = mx.Potential.soft_sphere(kappa=10, epsilon=0.000000001, r0=0.2, eta=2, tol=0.05, min=0, max=3)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_bs, Yolk, Cell)
mx.bind.types(pot_ss, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=0.05)

# bind it just like any other force
mx.bind.force(rforce, Cell)

yolk = Yolk(position=mx.Universe.center, velocity=[0., 0., 0.])

pts = [p * 0.5 * Yolk.radius + mx.Universe.center for p in mx.random_points(mx.PointsType.SolidSphere, count)]
pts = [p + [0, 0, 1.3 * Yolk.radius] for p in pts]

for p in pts:
    Cell(p)

# run the simulator interactive
mx.Simulator.run()
