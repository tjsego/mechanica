import mechanica as mx

# potential cutoff distance
cutoff = 8

receptor_count = 10000

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff, cells=[4, 4, 4], threads=8)


class NucleusType(mx.ParticleType):
    mass = 500000
    radius = 1


class ReceptorType(mx.ParticleType):
    mass = 0.2
    radius = 0.05
    target_temperature = 1
    # dynamics = mx.Overdamped


Nucleus = NucleusType.get()
Receptor = ReceptorType.get()

# locations of initial receptor positions
receptor_pts = [p * 5 + mx.Universe.center for p in mx.random_points(mx.PointsType.SolidSphere, receptor_count)]

pot_nr = mx.Potential.well(k=15, n=3, r0=7)
pot_rr = mx.Potential.morse(d=1, a=6, min=0.01, max=1, r0=0.3, shifted=False)

# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_rr, Receptor, Receptor)
mx.bind.types(pot_nr, Nucleus, Receptor)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = mx.Force.random(mean=0, std=3)
vtstat = mx.Force.random(mean=0, std=5)

# bind it just like any other force
mx.bind.force(tstat, Receptor)


n = Nucleus(position=mx.Universe.center, velocity=[0., 0., 0.])

for p in receptor_pts:
    Receptor(p)

# run the simulator interactive
mx.run()
