import mechanica as mx

# potential cutoff distance
cutoff = 8

receptor_count = 500

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff, cells=[4, 4, 4], integrator=mx.RUNGE_KUTTA_4)


class BigType(mx.ParticleType):
    mass = 500000
    radius = 3


class ReceptorType(mx.ParticleType):
    mass = 0.1
    radius = 0.1
    target_temperature = 1
    dynamics = mx.Overdamped


class VirusType(mx.ParticleType):
    mass = 1
    radius = 0.5
    target_temperature = 1
    dynamics = mx.Overdamped


Big, Receptor, Virus = BigType.get(), ReceptorType.get(), VirusType.get()

# locations of initial receptor positions
receptor_pts = [p * Big.radius + mx.Universe.center for p in mx.random_points(mx.PointsType.Sphere, receptor_count)]

pot_rr = mx.Potential.soft_sphere(kappa=0.02, epsilon=0, r0=0.5, eta=2, tol=0.05, min=0.01, max=4)
pot_vr = mx.Potential.soft_sphere(kappa=0.02, epsilon=0.1, r0=0.6, eta=4, tol=0.05, min=0.01, max=3)
pot_vb = mx.Potential.soft_sphere(kappa=5, epsilon=0, r0=4.8, eta=4, tol=0.05, min=3, max=5.5)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_rr, Receptor, Receptor)
mx.bind.types(pot_vr, Receptor, Virus)
mx.bind.types(pot_vb, Big, Virus)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = mx.Force.random(mean=0, std=0.1)
vtstat = mx.Force.random(mean=0, std=0.1)

# bind it just like any other force
mx.bind.force(tstat, Receptor)

mx.bind.force(vtstat, Virus)

b = Big(position=mx.Universe.center, velocity=[0., 0., 0.])

Virus(position=mx.Universe.center + [0, 0, Big.radius + 0.75])

harmonic = mx.Potential.harmonic(k=0.01 * 500, r0=Big.radius)

for p in receptor_pts:
    r = Receptor(p)
    mx.bind.particles(harmonic, b.part(), r.part())

# run the simulator interactive
mx.run()
