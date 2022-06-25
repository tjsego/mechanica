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


Yolk = YolkType.get()
Cell = CellType.get()

pot_bs = mx.Potential.morse(d=1, a=6, min=0, r0=3.0, max=9, shifted=False)
pot_ss = mx.Potential.morse(d=0.1, a=9, min=0, r0=0.3, max=0.6, shifted=False)

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


# import sphericalplot as sp
#
# plt = sp.SphericalPlot(Cell.items(), yolk.position)
#
# mx.on_time(invoke_method=plt.update, period=0.01)

# run the simulator interactive
mx.run()
