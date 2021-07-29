import mechanica as mx
import numpy as np

# dimensions of universe
dim = [30., 30., 30.]

mx.init(dim=dim,
        cutoff=10,
        integrator=mx.FORWARD_EULER,
        cells=[3, 3, 3],
        dt=0.001)


class AType(mx.ParticleType):
    radius = 1
    dynamics = mx.Newtonian
    mass = 20
    style = {"color": "MediumSeaGreen"}


A = AType.get()

p = mx.Potential.glj(e=0.1, m=3, max=3)
cp = mx.Potential.power(k=-100, alpha=0.1, r0=0, min=0.05, max=10, tol=0.001)

mx.bind.cuboid(p, A)
mx.bind.types(cp, A, A)

rforce = mx.Force.friction(0.01, 0, 100)

# bind it just like any other force
mx.bind.force(rforce, A)

c = mx.Cuboid.create(mx.Universe.center + [0, 0, 0], size=[25, 31, 5])

c.spin = [0.0, 8.5, 0.0]

# uniform random cube
positions = np.random.uniform(low=0, high=30, size=(2500, 3))

for p in positions:
    A(p, velocity=[0, 0, 0])

mx.run()
