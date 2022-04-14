import mechanica as mx

# potential cutoff distance
cutoff = 3

# number of particles
count = 6000

# number of time points we avg things
avg_pts = 3

# dimensions of universe
dim = [50., 50., 100.]

# new simulator
mx.init(dim=dim,
        cutoff=cutoff,
        integrator=mx.FORWARD_EULER,
        bc=mx.BOUNDARY_NONE,
        dt=0.001,
        max_distance=0.2,
        threads=8,
        cells=[5, 5, 5])

clump_radius = 8


class YolkType(mx.ParticleType):
    mass = 500000
    radius = 20
    frozen = True


class CellType(mx.ParticleType):
    mass = 10
    radius = 1.2
    target_temperature = 0
    dynamics = mx.Overdamped


Yolk = YolkType.get()
Cell = CellType.get()

total_height = 2 * Yolk.radius + 2 * clump_radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2 - 1.9 * clump_radius

pot_yc = mx.Potential.glj(e=500, r0=1, m=3, k=500, min=0.1, max=2 * Yolk.radius, tol=0.1)
pot_cc = mx.Potential.glj(e=50, r0=1, m=2, min=0.05, max=2.2 * Cell.radius)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_yc, Yolk, Cell)
mx.bind.types(pot_cc, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=50, duration=0.5)

# bind it just like any other force
mx.bind.force(rforce, Cell)

yolk = Yolk(position=mx.Universe.center + [0., 0., -yshift])

for i, p in enumerate(mx.random_points(mx.PointsType.SolidSphere, count)):
    pos = p * clump_radius + mx.Universe.center + [0., 0., cshift]
    Cell(position=pos)


# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
