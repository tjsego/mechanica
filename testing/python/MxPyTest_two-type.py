import mechanica as mx

# potential cutoff distance
cutoff = 8

count = 300

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff, windowless=True)


class BigType(mx.ParticleType):
    mass = 500000
    radius = 3


class SmallType(mx.ParticleType):
    mass = 0.1
    radius = 0.2
    target_temperature = 0


Big = BigType.get()
Small = SmallType.get()


pot_bs = mx.Potential.soft_sphere(kappa=100, epsilon=1, r0=3.2, eta=3, tol=0.1, min=0.1, max=8)

pot_ss = mx.Potential.soft_sphere(kappa=1, epsilon=0.1, r0=0.2, eta=2, tol=0.05, min=0.01, max=4)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_bs, Big, Small)
mx.bind.types(pot_ss, Small, Small)

Big(position=mx.Universe.center, velocity=[0., 0., 0.])

for p in mx.random_points(mx.PointsType.Disk, count):
    Small(p * 2.5 * Big.radius + mx.Universe.center + [0, 0, Big.radius + 1])

# run the simulator interactive
mx.step(100*mx.Universe.dt)


def test_pass():
    pass
