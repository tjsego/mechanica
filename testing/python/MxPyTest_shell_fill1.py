import mechanica as mx

# potential cutoff distance
cutoff = 10

# number of particles
count = 500

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
        cells=[5, 5, 5], 
        windowless=True)


class BigType(mx.ParticleType):
    mass = 500000
    radius = 20
    frozen = True


class SmallType(mx.ParticleType):
    mass = 10
    radius = 0.25
    target_temperature = 0
    dynamics = mx.Overdamped


Big = BigType.get()
Small = SmallType.get()

pot_yc = mx.Potential.glj(e=100, r0=5, m=3, k=500, min=0.1, max=1.5 * Big.radius, tol=0.1)
# pot_cc = mx.Potential.glj(e=0,   r0=2, mx=2, k=10,  min=0.05, max=1 * Big.radius)
pot_cc = mx.Potential.harmonic(r0=0, k=0.1, max=10)

# pot_yc = mx.Potential.glj(e=10, r0=1, mx=3, min=0.1, max=50*Small.radius, tol=0.1)
# pot_cc = mx.Potential.glj(e=100, r0=5, mx=2, min=0.05, max=0.5*Big.radius)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_yc, Big, Small)
mx.bind.types(pot_cc, Small, Small)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=100, duration=0.5)

# bind it just like any other force
mx.bind.force(rforce, Small)

yolk = Big(position=mx.Universe.center)


for p in mx.random_points(mx.PointsType.Sphere, count):
    pos = p * (Big.radius + Small.radius) + mx.Universe.center
    Small(position=pos)


# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
